"""
Multi-worker streaming pipeline with resume functionality:
  pre×N (CPU workers)  →  q_in   →  gpu (1 GPU proc)  →  q_out  →  post×M (CPU workers)

Single live-updating status panel shows progress and queue depths.
Automatically skips files that already exist in output directory.
"""

import argparse
import os
import pathlib
import queue
import sys
import time
from multiprocessing import Process, JoinableQueue, Event, Value, freeze_support, Manager

import cv2
import numpy as np
import tifffile
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from skimage.io import imread

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-worker nnUNet inference pipeline')

    # Paths
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input directory containing .tif files')
    parser.add_argument('--output-dir', type=str,
                        help='Output directory for processed files (default: input-dir with volumes → volumes_masked)')
    parser.add_argument('--model-dir', type=str, default='./models/nnUNetTrainerV2__nnUNetPlans__2d',
                        help='nnUNet model directory (default: ./models/nnUNetTrainerV2__nnUNetPlans__2d)')
    parser.add_argument('--temp-dir', type=str, default='/dev/shm/nnunet_tmp',
                        help='Temporary directory (default: /dev/shm/nnunet_tmp)')

    # Pipeline parameters
    parser.add_argument('--downscale-factor', type=int, default=4,
                        help='Image downscaling factor (default: 4)')
    parser.add_argument('--n-preproc', type=int, default=16,
                        help='Number of preprocessing workers (default: 16)')
    parser.add_argument('--n-postproc', type=int, default=16,
                        help='Number of postprocessing workers (default: 16)')
    parser.add_argument('--queue-size', '-q', type=int, default=512,
                        help='Maximum queue size for buffering (default: 512)')

    args = parser.parse_args()

    # Set default output directory if not provided
    if args.output_dir is None:
        input_path = pathlib.Path(args.input_dir)
        if 'volumes' in input_path.parts:
            # Replace 'volumes' with 'volumes_masked' in the path
            parts = list(input_path.parts)
            for i, part in enumerate(parts):
                if part == 'volumes':
                    parts[i] = 'volumes_masked'
                    break
            args.output_dir = str(pathlib.Path(*parts))
        else:
            # Fallback: append '_masked' to the input directory name
            args.output_dir = str(input_path.parent / (input_path.name + '_masked'))

    return args


def get_pending_files(args):
    """Returns list of files that need processing (not already in output dir)"""
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)

    all_files = list(input_dir.glob("*.tif"))
    if not output_dir.exists():
        return all_files

    # Clean up any leftover .tmp files from previous interrupted runs
    for tmp_file in output_dir.glob("*.tmp"):
        tmp_file.unlink()

    existing_outputs = {f.name for f in output_dir.glob("*.tif")}
    pending_files = [f for f in all_files if f.name not in existing_outputs]

    return pending_files


def create_status_panel(q_in_size, q_out_size, pre_count, gpu_count, post_count,
                        total_files, pending_files, skipped_files, failed_count, elapsed_time, start_post_count,
                        avg_gpu_time):
    # Calculate speeds
    if elapsed_time > 0:
        current_processed = post_count - start_post_count
        files_per_sec = current_processed / elapsed_time
        remaining = pending_files - post_count - failed_count
        eta_sec = remaining / files_per_sec if files_per_sec > 0 else 0

        if eta_sec >= 60:
            eta_display = f"{int(eta_sec / 60)}m"
        else:
            eta_display = f"{int(eta_sec)}s"
    else:
        files_per_sec = 0
        eta_display = "0s"

    status_text = Text()
    status_text.append("Queues: ", style="bold")
    status_text.append(f"in={q_in_size} ", style="cyan")
    status_text.append(f"out={q_out_size} ", style="blue")
    status_text.append("| Progress: ", style="bold")
    status_text.append(f"pre={pre_count} ", style="cyan")
    status_text.append(f"gpu={gpu_count} ", style="green")
    status_text.append(f"saved={post_count} ", style="blue")
    if failed_count > 0:
        status_text.append(f"failed={failed_count} ", style="red")
    status_text.append(f"| Total: {post_count + skipped_files}/{total_files} ", style="white")
    status_text.append(f"| Speed: {files_per_sec:.1f}/s ", style="yellow")
    if avg_gpu_time > 0:
        status_text.append(f"GPU: {avg_gpu_time * 1000:.0f}ms/img ", style="green")
    if files_per_sec > 0:
        status_text.append(f"ETA: {eta_display}", style="magenta")

    return Panel(status_text, expand=False, padding=(0, 1))


def pad_to_stride(img: np.ndarray, stride_xy: tuple[int, int]):
    h, w = img.shape
    sx, sy = stride_xy
    ph = (sx - h % sx) % sx
    pw = (sy - w % sy) % sy
    if ph or pw: img = np.pad(img, ((0, ph), (0, pw)), mode='edge')
    return img, (h, w)


def pre_worker(file_queue: JoinableQueue, q_in: JoinableQueue, stop: Event, pre_counter: Value, failed_files: list,
               args):
    """Single preprocessing worker - reads from file_queue, outputs to q_in"""
    try:
        while not stop.is_set():
            try:
                tif_path = file_queue.get(timeout=1)
                if tif_path is None:  # sentinel
                    break
            except queue.Empty:
                continue

            try:
                img = imread(tif_path)
                original_dtype = img.dtype
                downscaled_shape = (
                    img.shape[0] // args.downscale_factor,
                    img.shape[1] // args.downscale_factor
                )

                img_ds = cv2.resize(img, (downscaled_shape[1], downscaled_shape[0]),
                                    interpolation=cv2.INTER_AREA).astype(original_dtype)

                q_in.put((tif_path.name, img_ds, img.shape, img.dtype.str))
                with pre_counter.get_lock():
                    pre_counter.value += 1

            except Exception as read_error:
                failed_files.append(f"{tif_path.name}: {str(read_error)}")
                continue  # skip this file, move to next

    except Exception as e:
        console.print(f"[red]PRE WORKER ERROR: {e}[/red]")
        raise


def gpu_proc(q_in: JoinableQueue, q_out: JoinableQueue, stop: Event, gpu_counter: Value, gpu_times: list,
             gpu_ready: Event, args):
    try:
        console.print("[yellow]Initializing GPU and loading model...[/yellow]")
        device = torch.device('cuda')
        predictor = nnUNetPredictor(tile_step_size=1.0,  # no overlap = fastest (slight quality loss)
                                    use_gaussian=True,
                                    use_mirroring=False,  # test-time aug; True = slower but tiny accuracy gain
                                    perform_everything_on_device=True,
                                    device=device, verbose=False)

        ckpt = next(p for p in (pathlib.Path(args.model_dir) / "fold_0").glob("checkpoint_*.pth"))
        predictor.initialize_from_trained_model_folder(args.model_dir, use_folds=(0,), checkpoint_name=ckpt.name)
        predictor.network.to(device).half()

        console.print("[yellow]Compiling model with torch.compile...[/yellow]")
        predictor.network = torch.compile(predictor.network)

        console.print("[green]✓ GPU ready! Starting inference...[/green]")
        gpu_ready.set()  # Signal that GPU is ready

        pk = np.array(predictor.configuration_manager.pool_op_kernel_sizes)
        stride = tuple(int(np.prod(pk[:, i])) for i in range(2))

        while True:
            if stop.is_set(): break
            try:
                item = q_in.get(timeout=1)
                if item is None: break
            except queue.Empty:
                continue

            fname, img_ds, orig_shape, dtype_str = item
            img_ds, crop_sz = pad_to_stride(img_ds, stride)

            img_ds = np.clip(img_ds, 0, 65000).astype(np.float32) / 65000.0
            tens = torch.from_numpy(img_ds).half().unsqueeze(0).unsqueeze(0).to(device)

            # Time the GPU inference
            start_time = time.time()
            with torch.no_grad():
                logits = torch.sigmoid(predictor.network(tens))[0, 0].cpu().numpy()
            gpu_time = time.time() - start_time
            gpu_times.append(gpu_time)

            logits = logits[:crop_sz[0], :crop_sz[1]]

            q_out.put((fname, logits, orig_shape, dtype_str))
            with gpu_counter.get_lock():
                gpu_counter.value += 1

    except Exception as e:
        console.print(f"[red]GPU ERROR: {e}[/red]")
        gpu_ready.set()  # Signal even if failed
        raise
    finally:
        # Signal post workers to stop
        for _ in range(args.n_postproc):
            q_out.put(None)


def post_worker(q_out: JoinableQueue, stop: Event, post_counter: Value, failed_files: list, args):
    """Single postprocessing worker - reads from q_out"""
    try:
        out_dir = pathlib.Path(args.output_dir)

        while not stop.is_set():
            try:
                item = q_out.get(timeout=1)
                if item is None: break
            except queue.Empty:
                continue

            fname, logits, orig_shape, dtype_str = item

            # Double-check file doesn't exist (race condition protection)
            output_path = out_dir / fname
            if output_path.exists():
                with post_counter.get_lock():
                    post_counter.value += 1
                continue

            try:
                mask = (logits < 0.5).astype(np.uint8)
                H, W = orig_shape
                mask_us = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                img = tifffile.imread(pathlib.Path(args.input_dir) / fname).astype(np.dtype(dtype_str))

                # Atomic write: temp file first, then rename
                temp_path = output_path.with_suffix('.tmp')
                tifffile.imwrite(temp_path, img * mask_us)
                temp_path.rename(output_path)  # atomic operation

                with post_counter.get_lock():
                    post_counter.value += 1

            except Exception as save_error:
                failed_files.append(f"{fname}: {str(save_error)}")
                continue  # skip this file, move to next

    except Exception as e:
        console.print(f"[red]POST WORKER ERROR: {e}[/red]")
        raise


def main():
    freeze_support()
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get file counts for resume functionality
    all_input_files = list(pathlib.Path(args.input_dir).glob("*.tif"))
    pending_files = get_pending_files(args)

    total_files = len(all_input_files)
    skipped_files = total_files - len(pending_files)
    pending_count = len(pending_files)

    if skipped_files > 0:
        console.print(f"[yellow]Resuming: {skipped_files} files already processed, {pending_count} remaining[/yellow]")

    if pending_count == 0:
        console.print("[green]✓ All files already processed![/green]")
        return 0

    # Shared counters and failed files list
    pre_counter = Value('i', 0)
    gpu_counter = Value('i', 0)
    post_counter = Value('i', 0)

    manager = Manager()
    failed_files = manager.list()  # shared list for failed file names
    gpu_times = manager.list()  # shared list for GPU timing

    # Queues
    file_queue = JoinableQueue()  # feeds file paths to pre workers
    q_in = JoinableQueue(maxsize=args.queue_size)  # pre → gpu
    q_out = JoinableQueue(maxsize=args.queue_size)  # gpu → post
    stop = Event()
    gpu_ready = Event()  # signals when GPU initialization is complete

    # Populate file queue
    for tif_path in pending_files:
        file_queue.put(tif_path)

    # Add sentinels for pre workers
    for _ in range(args.n_preproc):
        file_queue.put(None)

    # Create worker processes
    pre_workers = [
        Process(target=pre_worker, args=(file_queue, q_in, stop, pre_counter, failed_files, args), name=f"PRE-{i}")
        for i in range(args.n_preproc)
    ]

    gpu = Process(target=gpu_proc, args=(q_in, q_out, stop, gpu_counter, gpu_times, gpu_ready, args), name="GPU")

    post_workers = [
        Process(target=post_worker, args=(q_out, stop, post_counter, failed_files, args), name=f"POST-{i}")
        for i in range(args.n_postproc)
    ]

    all_processes = pre_workers + [gpu] + post_workers

    console.print(f"[green]Starting {args.n_preproc} pre + 1 GPU + {args.n_postproc} post workers[/green]")
    for p in all_processes:
        p.start()

    # Wait for GPU initialization to complete before starting the live panel
    gpu_ready.wait()

    start_time = time.time()
    start_post_count = post_counter.value

    try:
        with Live(console=console, refresh_per_second=4) as live:
            while any(p.is_alive() for p in all_processes):
                elapsed = time.time() - start_time
                failed_count = len(failed_files)

                # Calculate average GPU time
                avg_gpu_time = sum(gpu_times[-10:]) / len(gpu_times[-10:]) if gpu_times else 0

                panel = create_status_panel(
                    q_in.qsize(),
                    q_out.qsize(),
                    pre_counter.value,
                    gpu_counter.value,
                    post_counter.value,
                    total_files,
                    pending_count,
                    skipped_files,
                    failed_count,
                    elapsed,
                    start_post_count,
                    avg_gpu_time
                )
                live.update(panel)

                # Check if we're done (all files processed or failed)
                completed_files = post_counter.value + failed_count
                if completed_files >= pending_count and not any(p.is_alive() for p in pre_workers):
                    stop.set()
                    break

                time.sleep(0.25)

                # Check for crashes
                for p in all_processes:
                    if not p.is_alive() and p.exitcode not in (0, None):
                        stop.set()
                        raise RuntimeError(f"{p.name} crashed with exitcode {p.exitcode}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted[/yellow]")
        stop.set()
    finally:
        for p in all_processes:
            p.join()

        final_processed = skipped_files + post_counter.value
        failed_count = len(failed_files)

        console.print(f"\n[green]✓ Pipeline complete: {final_processed}/{total_files} files processed[/green]")

        if failed_count > 0:
            console.print(f"[red]✗ {failed_count} files failed:[/red]")
            for failed_file in failed_files:
                console.print(f"  [red]{failed_file}[/red]")


if __name__ == "__main__":
    sys.exit(main())
