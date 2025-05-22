#!/usr/bin/env python3

import argparse
import glob
import os
import shutil
import sys

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


class Config:
    def __init__(self):
        self.input_dir = os.path.expanduser("~/scrollprize/data/scroll4.volpkg/volumes/20231117161658")
        self.output_dir = os.path.expanduser("~/scrollprize/data/scroll4.volpkg/volumes_masked/20231117161658")
        self.model_dir = "./models/nnUNetTrainerV2__nnUNetPlans__2d"
        self.downscale_factor = 4
        self.threshold = 0.5
        self.batch_size = 50  # Process files in batches
        self.temp_dir = "./tmp"

        self._setup_nnunet_env()

    def _setup_nnunet_env(self):
        """Set up nnUNet environment variables and create directories"""
        base_temp = os.path.abspath(os.path.join(self.temp_dir, "nnunet_processing"))
        os.environ["nnUNet_raw"] = os.path.join(base_temp, "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = os.path.join(base_temp, "nnUNet_preprocessed")
        os.environ["nnUNet_results"] = os.path.join(base_temp, "nnUNet_results")

        dataset_name = "Dataset082_scrollmask2"

        directories = [
            self.temp_dir,
            base_temp,
            os.environ["nnUNet_raw"],
            os.environ["nnUNet_preprocessed"],
            os.environ["nnUNet_results"],
            os.path.join(os.environ["nnUNet_raw"], dataset_name),
            os.path.join(os.environ["nnUNet_raw"], dataset_name, "imagesTs"),
            os.path.join(os.environ["nnUNet_results"], dataset_name),
        ]

        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)


config = Config()


class BatchScrollMaskProcessor:
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

            device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
            console.print(f"[blue]Loading nnUNet model on {device}...[/blue]")

            self.predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True  # Changed to match working script
            )

            self._fix_trainer_name()

            # Load model weights
            for checkpoint in ["checkpoint_best.pth", "checkpoint_final.pth"]:
                try:
                    print(f"Loading {checkpoint}...")
                    self.predictor.initialize_from_trained_model_folder(
                        self.model_dir,
                        use_folds=(0,),
                        checkpoint_name=checkpoint
                    )
                    print(f"✓ Loaded {checkpoint}")
                    break
                except Exception as e:
                    print(f"Failed to load {checkpoint}: {e}")
                    continue
            else:
                raise RuntimeError("Could not load any checkpoint")

        except ImportError:
            raise ImportError("nnunetv2 not installed. Run: pip install nnunetv2")

    def _fix_trainer_name(self):
        """Fix trainer name in checkpoint if needed"""
        checkpoint_names = ["checkpoint_best.pth", "checkpoint_final.pth"]
        ckpt_path = None

        # Look in fold_0/ first
        for name in checkpoint_names:
            p = os.path.join(self.model_dir, "fold_0", name)
            if os.path.exists(p):
                ckpt_path = p
                break

        # Fallback to model root
        if ckpt_path is None:
            for name in checkpoint_names:
                p = os.path.join(self.model_dir, name)
                if os.path.exists(p):
                    ckpt_path = p
                    break

        if ckpt_path is None:
            console.print("[yellow]Warning: No checkpoint found to fix trainer name[/yellow]")
            return

        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            orig = checkpoint.get("trainer_name", "")
            if orig != "nnUNetTrainer":
                console.print(f"[yellow]Fixing trainer_name '{orig}' → 'nnUNetTrainer'[/yellow]")
                backup_path = ckpt_path + ".backup"
                if not os.path.exists(backup_path):
                    shutil.copy2(ckpt_path, backup_path)
                checkpoint["trainer_name"] = "nnUNetTrainer"
                torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False)
            else:
                console.print(f"[green]Trainer name OK: {orig}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not fix trainer name: {e}[/yellow]")

    def process_batch(self, input_files, output_files):
        """Process a batch of files similar to the working script"""
        # Prepare temporary directory for this batch
        batch_id = os.path.basename(input_files[0]).split('.')[0]
        temp_input_dir = os.path.join(os.environ["nnUNet_raw"], "Dataset082_scrollmask2", "imagesTs")
        temp_output_dir = os.path.join(os.environ["nnUNet_results"], "Dataset082_scrollmask2", f"batch_{batch_id}")

        os.makedirs(temp_input_dir, exist_ok=True)
        os.makedirs(temp_output_dir, exist_ok=True)

        # Clean up any existing files in temp directories
        for f in os.listdir(temp_input_dir):
            os.remove(os.path.join(temp_input_dir, f))

        # Process all files in batch
        processed_files = []
        file_mapping = {}  # Map temp names to original names

        for input_path, output_path in zip(input_files, output_files):
            if os.path.exists(output_path):
                console.print(f"[yellow]Skipping existing: {os.path.basename(output_path)}[/yellow]")
                continue

            # Load and downscale
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

            # Save with proper naming
            base_filename = os.path.splitext(os.path.basename(input_path))[0]
            nnunet_filename = f"{base_filename}_0000.tif"
            nnunet_path = os.path.join(temp_input_dir, nnunet_filename)

            imsave(nnunet_path, downscaled_image, check_contrast=False)

            processed_files.append((input_path, output_path, base_filename, original_image, original_dtype))
            file_mapping[base_filename] = (input_path, output_path)

        if not processed_files:
            return 0  # Nothing to process

        # Run inference on the entire directory (like the working script)
        console.print(f"[blue]Running batch inference on {len(processed_files)} files...[/blue]")

        self.predictor.predict_from_files(
            temp_input_dir,  # Directory path, not list
            temp_output_dir,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,  # Match working script
            num_processes_segmentation_export=2,  # Match working script
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0
        )

        # Process results
        success_count = 0
        for input_path, output_path, base_filename, original_image, original_dtype in processed_files:
            # Find the prediction file
            prediction_path = None
            possible_names = [
                f"{base_filename}.tif",
                f"{base_filename}.nii.gz",
                f"{base_filename}_0000.tif",
                f"{base_filename}_0000.nii.gz"
            ]

            for name in possible_names:
                path = os.path.join(temp_output_dir, name)
                if os.path.exists(path):
                    prediction_path = path
                    break

            if not prediction_path:
                console.print(f"[red]No prediction found for {base_filename}[/red]")
                continue

            try:
                # Load prediction
                if prediction_path.endswith('.nii.gz'):
                    pred_nii = nib.load(prediction_path)
                    prediction = pred_nii.get_fdata()
                else:
                    prediction = imread(prediction_path)

                prediction = np.squeeze(prediction)

                # Convert to binary mask
                binary_mask = (prediction >= self.threshold).astype(np.uint8)

                # Upscale mask
                upscaled_mask = resize(
                    binary_mask,
                    original_image.shape,
                    order=0,
                    preserve_range=True,
                    anti_aliasing=False
                ).astype(np.uint8)

                # Apply mask
                masked_image = original_image * upscaled_mask

                # Save result
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                imsave(output_path, masked_image.astype(original_dtype), check_contrast=False)

                console.print(f"[green]✓ Processed: {os.path.basename(output_path)}[/green]")
                success_count += 1

            except Exception as e:
                console.print(f"[red]✗ Error processing {base_filename}: {e}[/red]")

        # Clean up temp files
        shutil.rmtree(temp_output_dir, ignore_errors=True)
        for f in os.listdir(temp_input_dir):
            os.remove(os.path.join(temp_input_dir, f))

        return success_count


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
        description='Process TIF slices with nnUNet masking pipeline (batch mode)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_scroll_masks_batch.py
  python process_scroll_masks_batch.py --batch-size 100
  python process_scroll_masks_batch.py --input ./volumes/20231117161658 --output ./volumes_masked/20231117161658
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
    parser.add_argument('--batch-size', type=int, default=config.batch_size,
                        help=f'Number of files to process per batch (default: {config.batch_size})')
    parser.add_argument('--resume', action='store_true',
                        help='Resume processing (skip existing output files)')

    args = parser.parse_args()

    console.print(Panel("[bold blue]Batch Scroll Mask Processor[/bold blue]", title="Starting"))

    # Validate inputs
    if not os.path.exists(args.input):
        console.print(f"[red]Error: Input directory {args.input} does not exist![/red]")
        return 1

    if not os.path.exists(args.model):
        console.print(f"[red]Error: Model directory {args.model} does not exist![/red]")
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

    # Display configuration
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Input Directory", args.input)
    table.add_row("Output Directory", args.output)
    table.add_row("Model Directory", args.model)
    table.add_row("Files to Process", str(len(file_pairs)))
    table.add_row("Batch Size", str(args.batch_size))
    table.add_row("Downscale Factor", str(args.downscale))
    table.add_row("Threshold", str(args.threshold))
    table.add_row("Resume Mode", "Yes" if args.resume else "No")

    console.print(table)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize processor
    processor = BatchScrollMaskProcessor(args.model, args.threshold, args.downscale)

    # Process in batches
    total_processed = 0
    total_batches = (len(file_pairs) + args.batch_size - 1) // args.batch_size

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
    ) as progress:

        task = progress.add_task(f"[cyan]Processing {total_batches} batches...", total=len(file_pairs))

        for i in range(0, len(file_pairs), args.batch_size):
            batch = file_pairs[i:i + args.batch_size]
            batch_input_files = [pair[0] for pair in batch]
            batch_output_files = [pair[1] for pair in batch]

            batch_num = i // args.batch_size + 1
            console.print(f"\n[blue]Processing batch {batch_num}/{total_batches} ({len(batch)} files)...[/blue]")

            processed = processor.process_batch(batch_input_files, batch_output_files)
            total_processed += processed

            progress.update(task, advance=len(batch))

    # Final summary
    console.print(Panel(f"[green]Successfully processed {total_processed} files![/green]", title="Complete"))
    console.print(f"[green]Masked images saved to: {args.output}[/green]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
