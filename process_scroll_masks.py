#!/usr/bin/env python3

import argparse
import glob
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from skimage.io import imread, imsave
from skimage.transform import resize

console = Console()


# Configuration
class Config:
    def __init__(self):
        self.input_dir = os.path.expanduser("~/scrollprize/data/scroll4.volpkg/volumes/20231117161658")
        self.output_dir = os.path.expanduser("~/scrollprize/data/scroll4.volpkg/volumes_masked/20231117161658")
        self.model_dir = "./models/nnUNetTrainerV2__nnUNetPlans__2d"
        self.downscale_factor = 4
        self.threshold = 0.5
        self.n_workers = None
        self.temp_dir = "./tmp"


config = Config()


class ScrollMaskProcessor:
    def __init__(self, model_dir, threshold=0.5, downscale_factor=4):
        self.model_dir = model_dir
        self.threshold = threshold
        self.downscale_factor = downscale_factor
        self.predictor = None
        self._setup_model()

    def _setup_model(self):
        """Initialize the nnUNet predictor once"""
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            console.print(f"[blue]Loading nnUNet model on {device}...[/blue]")

            self.predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False
            )

            # Fix trainer name in checkpoint if needed
            self._fix_trainer_name()

            # Load model weights
            for checkpoint in ["checkpoint_best.pth", "checkpoint_final.pth"]:
                try:
                    with console.status(f"[bold yellow]Loading {checkpoint}..."):
                        self.predictor.initialize_from_trained_model_folder(
                            self.model_dir,
                            use_folds=(0,),
                            checkpoint_name=checkpoint
                        )
                    console.print(f"[green]✓[/green] Loaded {checkpoint}")
                    break
                except Exception as e:
                    console.print(f"[yellow]Failed to load {checkpoint}: {e}[/yellow]")
                    continue
            else:
                raise RuntimeError("Could not load any checkpoint")

        except ImportError:
            raise ImportError("nnunetv2 not installed. Run: pip install nnunetv2")

    def _fix_trainer_name(self):
        """Fix trainer name in checkpoint if needed"""
        for checkpoint_name in ["checkpoint_best.pth", "checkpoint_final.pth"]:
            checkpoint_path = None

            # Check fold_0 directory first
            fold_path = os.path.join(self.model_dir, "fold_0", checkpoint_name)
            if os.path.exists(fold_path):
                checkpoint_path = fold_path
            else:
                # Fallback to model root
                root_path = os.path.join(self.model_dir, checkpoint_name)
                if os.path.exists(root_path):
                    checkpoint_path = root_path

            if checkpoint_path:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    if checkpoint.get("trainer_name") != "nnUNetTrainer":
                        backup_path = checkpoint_path + ".backup"
                        if not os.path.exists(backup_path):
                            shutil.copy2(checkpoint_path, backup_path)
                        checkpoint["trainer_name"] = "nnUNetTrainer"
                        torch.save(checkpoint, checkpoint_path, _use_new_zipfile_serialization=False)
                        console.print(f"[yellow]Fixed trainer name in {checkpoint_name}[/yellow]")
                except Exception as e:
                    console.print(f"[red]Warning: Could not fix trainer name in {checkpoint_name}: {e}[/red]")
                break

    def process_single_file(self, input_path, output_path):
        """Process a single TIF file through the complete pipeline"""
        try:
            # Skip if output already exists
            if os.path.exists(output_path):
                return True, f"Skipped (already exists): {os.path.basename(input_path)}"

            # Step 1: Load and downscale
            original_image = imread(input_path)
            original_dtype = original_image.dtype

            downscaled_shape = (
                original_image.shape[0] // self.downscale_factor,
                original_image.shape[1] // self.downscale_factor
            )

            downscaled_image = resize(
                original_image,
                downscaled_shape,
                preserve_range=True,
                anti_aliasing=True
            ).astype(original_dtype)

            # Step 2: Create temporary file for inference
            temp_dir = Path(config.temp_dir) / "processing"
            temp_dir.mkdir(parents=True, exist_ok=True)

            temp_input = temp_dir / f"temp_{os.getpid()}_{os.path.basename(input_path)}"
            temp_output = temp_dir / f"temp_{os.getpid()}_output.nii.gz"

            # Save downscaled image temporarily
            imsave(str(temp_input), downscaled_image, check_contrast=False)

            # Step 3: Run inference
            self.predictor.predict_from_files(
                [str(temp_input)],
                str(temp_dir),
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
                folder_with_segs_from_prev_stage=None,
                num_parts=1,
                part_id=0
            )

            # Find the prediction output
            prediction_files = list(temp_dir.glob("*.nii.gz"))
            if not prediction_files:
                raise RuntimeError("No prediction output found")

            prediction_path = prediction_files[0]

            # Step 4: Load prediction and apply threshold
            pred_nii = nib.load(str(prediction_path))
            prediction = pred_nii.get_fdata()
            prediction = np.squeeze(prediction)

            # Convert to binary mask
            binary_mask = (prediction >= self.threshold).astype(np.uint8)

            # Step 5: Upscale mask to original size
            upscaled_mask = resize(
                binary_mask,
                original_image.shape,
                order=0,  # Nearest neighbor for binary mask
                preserve_range=True,
                anti_aliasing=False
            ).astype(np.uint8)

            # Step 6: Apply mask to original image
            masked_image = original_image * upscaled_mask

            # Step 7: Save result
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            imsave(output_path, masked_image.astype(original_dtype), check_contrast=False)

            # Clean up temporary files
            temp_input.unlink(missing_ok=True)
            for temp_file in temp_dir.glob(f"temp_{os.getpid()}*"):
                temp_file.unlink(missing_ok=True)

            return True, f"Processed: {os.path.basename(input_path)}"

        except Exception as e:
            return False, f"Error processing {os.path.basename(input_path)}: {str(e)}"


def process_file_wrapper(args):
    """Wrapper function for multiprocessing"""
    input_path, output_path, model_dir, threshold, downscale_factor = args

    # Create processor instance in each worker
    processor = ScrollMaskProcessor(model_dir, threshold, downscale_factor)
    return processor.process_single_file(input_path, output_path)


def get_file_pairs(input_dir, output_dir):
    """Get input-output file pairs"""
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))

    if not input_files:
        raise ValueError(f"No TIF files found in {input_dir}")

    file_pairs = []
    for input_path in input_files:
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        file_pairs.append((input_path, output_path))

    return file_pairs


def main():
    parser = argparse.ArgumentParser(
        description='Process TIF slices with nnUNet masking pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_scroll_masks.py
  python process_scroll_masks.py --input ./volumes/20231117161658 --output ./volumes_masked/20231117161658
  python process_scroll_masks.py --workers 4 --threshold 0.3
        """
    )

    parser.add_argument('--input', type=str, default=config.input_dir,
                        help=f'Input directory with TIF files (default: {config.input_dir})')
    parser.add_argument('--output', type=str, default=config.output_dir,
                        help=f'Output directory for masked TIF files (default: {config.output_dir})')
    parser.add_argument('--model', type=str, default=config.model_dir,
                        help=f'Path to nnUNet model directory (default: {config.model_dir})')
    parser.add_argument('--downscale', type=int, default=config.downscale_factor,
                        help=f'Downscale factor (default: {config.downscale_factor})')
    parser.add_argument('--threshold', type=float, default=config.threshold,
                        help=f'Mask threshold (default: {config.threshold})')
    parser.add_argument('--workers', type=int, default=config.n_workers,
                        help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume processing (skip existing output files)')

    args = parser.parse_args()

    console.print(Panel("[bold blue]Scroll Mask Processor[/bold blue]", title="Starting"))

    # Validate inputs
    if not os.path.exists(args.input):
        console.print(f"[red]Error: Input directory {args.input} does not exist![/red]")
        return 1

    if not os.path.exists(args.model):
        console.print(f"[red]Error: Model directory {args.model} does not exist![/red]")
        console.print("[yellow]Please run setup.py first or check the model path.[/yellow]")
        return 1

    # Get file pairs
    try:
        file_pairs = get_file_pairs(args.input, args.output)
        console.print(f"[green]Found {len(file_pairs)} TIF files to process[/green]")
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1

    # Filter existing files if resuming
    if args.resume:
        original_count = len(file_pairs)
        file_pairs = [(inp, out) for inp, out in file_pairs if not os.path.exists(out)]
        skipped = original_count - len(file_pairs)
        if skipped > 0:
            console.print(f"[yellow]Resuming: skipped {skipped} already processed files[/yellow]")

    if not file_pairs:
        console.print("[green]No files to process![/green]")
        return 0

    # Setup workers
    n_workers = args.workers or min(multiprocessing.cpu_count(), len(file_pairs))

    # Display configuration
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Input Directory", args.input)
    table.add_row("Output Directory", args.output)
    table.add_row("Model Directory", args.model)
    table.add_row("Files to Process", str(len(file_pairs)))
    table.add_row("Workers", str(n_workers))
    table.add_row("Downscale Factor", str(args.downscale))
    table.add_row("Threshold", str(args.threshold))
    table.add_row("Resume Mode", "Yes" if args.resume else "No")

    console.print(table)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Prepare arguments for workers
    worker_args = [
        (input_path, output_path, args.model, args.threshold, args.downscale)
        for input_path, output_path in file_pairs
    ]

    # Process files
    successful = 0
    failed = 0
    errors = []

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
    ) as progress:

        if n_workers == 1:
            # Sequential processing
            task = progress.add_task("[cyan]Processing files...", total=len(worker_args))

            for args_tuple in worker_args:
                success, message = process_file_wrapper(args_tuple)
                if success:
                    successful += 1
                    progress.console.print(f"[green]✓[/green] {message}")
                else:
                    failed += 1
                    errors.append(message)
                    progress.console.print(f"[red]✗[/red] {message}")
                progress.update(task, advance=1)
        else:
            # Parallel processing
            task = progress.add_task("[cyan]Processing files...", total=len(worker_args))

            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(process_file_wrapper, args_tuple): args_tuple
                           for args_tuple in worker_args}

                for future in as_completed(futures):
                    try:
                        success, message = future.result()
                        if success:
                            successful += 1
                            if not message.startswith("Skipped"):
                                progress.console.print(f"[green]✓[/green] {message}")
                        else:
                            failed += 1
                            errors.append(message)
                            progress.console.print(f"[red]✗[/red] {message}")
                    except Exception as e:
                        failed += 1
                        args_tuple = futures[future]
                        error_msg = f"Unexpected error processing {os.path.basename(args_tuple[0])}: {e}"
                        errors.append(error_msg)
                        progress.console.print(f"[red]✗[/red] {error_msg}")

                    progress.update(task, advance=1)

    # Final summary
    summary_table = Table(title="Processing Summary")
    summary_table.add_column("Result", style="cyan")
    summary_table.add_column("Count", justify="right", style="magenta")

    summary_table.add_row("Successfully Processed", str(successful))
    summary_table.add_row("Failed", str(failed))
    summary_table.add_row("Total", str(successful + failed))

    console.print(summary_table)

    if successful > 0:
        console.print(Panel(f"[green]Masked images saved to: {args.output}[/green]", title="Success"))

    if errors:
        console.print(
            Panel("\n".join(errors[:10]) + (f"\n... and {len(errors) - 10} more errors" if len(errors) > 10 else ""),
                  title="[red]Errors[/red]"))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
