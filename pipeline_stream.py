#!/usr/bin/env python3
"""
Streaming Scroll-Mask pipeline
    – CPU stage A:  read → down-scale
    – GPU stage B:  nnUNet forward pass
    – CPU stage C:  up-scale → apply mask → write
All three stages run concurrently and keep the GPU busy.
"""

import pathlib, sys, cv2, tifffile, numpy as np, torch, queue, time, os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, JoinableQueue, freeze_support
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from rich.console import Console
from config import Config as C

console = Console()


# ---------- helpers ----------------------------------------------------------

def out_path(in_path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(C.output_dir) / in_path.name


# ---------- Stage A:  read + down-scale -------------------------------------
def pre_worker(path: pathlib.Path):
    img = tifffile.imread(path)
    h, w = img.shape
    img_ds = cv2.resize(
        img,
        (w // C.downscale_factor, h // C.downscale_factor),
        interpolation=cv2.INTER_AREA,
    )
    return (
        path.name,
        img_ds,  # maybe padded
        img.shape,  # full-res shape → for final up-scale
        img.dtype.str,
    )


# ---------- Stage B:  GPU inference -----------------------------------------

def pad_to_stride(img: np.ndarray, stride_hw: tuple[int, int]):
    """Pad H and W up to a multiple of stride_hw (bottom/right edge)."""
    h, w = img.shape
    sh, sw = stride_hw
    pad_h = (sh - h % sh) % sh
    pad_w = (sw - w % sw) % sw
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
    return img, (h, w)              # original size for later cropping

def gpu_worker(q_in: JoinableQueue, q_out: JoinableQueue, stop_token):
    device = torch.device('cuda', 0)
    torch.set_grad_enabled(False)

    predictor = nnUNetPredictor(tile_step_size=C.tile_step_size,
                                use_gaussian=True,
                                use_mirroring=C.use_mirroring,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=False)

    # pick first existing checkpoint
    ckpt = None
    for name in ("checkpoint_best.pth", "checkpoint_final.pth"):
        cand = pathlib.Path(C.model_dir) / "fold_0" / name
        if cand.exists(): ckpt = name; break
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found in model_dir")

    predictor.initialize_from_trained_model_folder(C.model_dir, use_folds=(0,), checkpoint_name=ckpt)
    predictor.network.to(device).half()

    import numpy as np
    try:
        pool_kernels = np.array(predictor.network.pool_op_kernel_sizes)
    except AttributeError:
        pool_kernels = np.array(predictor.configuration_manager.pool_op_kernel_sizes)

    # product of pool kernel sizes per axis, e.g. [32, 32] or [32, 16]
    stride_hw = tuple(int(np.prod(pool_kernels[:, i])) for i in range(2))
    print("network stride H,W =", stride_hw)

    while True:
        item = q_in.get()
        if item is stop_token:
            q_in.task_done();
            break

        fname, img_ds, orig_shape, dtype_str = item
        print("GPU got", fname, img_ds.shape, flush=True)

        # pad to network stride
        img_ds, ds_orig_shape = pad_to_stride(img_ds, stride_hw)

        img_tensor = torch.from_numpy(img_ds).unsqueeze(0).unsqueeze(0) \
            .to(device).half()

        with torch.no_grad():
            logits = torch.sigmoid(predictor.network(img_tensor))[0, 0] \
                         .cpu().numpy()[: ds_orig_shape[0], : ds_orig_shape[1]]

        print("GPU done", fname, flush=True)

        q_out.put((fname, logits, orig_shape, dtype_str))
        q_in.task_done()


# ---------- Stage C:  up-scale, mask, write ---------------------------------

def post_worker(packet):
    try:
        fname, logits, orig_shape, dtype_str = packet
        mask = (logits >= C.threshold).astype(np.uint8)

        H, W = orig_shape
        mask_us = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        in_full = pathlib.Path(C.input_dir) / fname
        out_full = out_path(in_full)

        if out_full.exists():  # resume support
            return 1

        img = tifffile.imread(in_full).astype(np.dtype(dtype_str))

        try:
            tifffile.imwrite(out_full, img * mask_us, compression="deflate")  # ← use a codec that exists
        except Exception as e:
            print("✗ post_worker error on", fname, "→", e, flush=True)
            return 0
        print("✓ saved", out_full, flush=True)
        return 1
    except Exception as e:
        return f"ERROR {fname}: {e}"


# ---------- main -------------------------------------------------------------

def main():
    freeze_support()  # needed on Windows

    os.makedirs(C.output_dir, exist_ok=True)

    input_paths = sorted(pathlib.Path(C.input_dir).glob("*.tif"))
    if not input_paths:
        console.print(f"[red]No .tif files in {C.input_dir}[/red]")
        return 1

    q_in = JoinableQueue(maxsize=C.queue_size)
    q_out = JoinableQueue(maxsize=C.queue_size)
    STOP = object()

    console.print(f"[cyan]Starting pipeline on {len(input_paths)} slices …[/cyan]")

    # start GPU process
    gpu_p = Process(target=gpu_worker, args=(q_in, q_out, STOP), daemon=True)
    gpu_p.start()

    processed = 0
    t0 = time.time()

    # pool for post-processing
    with ProcessPoolExecutor(max_workers=C.n_postproc) as post_pool, \
            ProcessPoolExecutor(max_workers=C.n_preproc) as pre_pool:

        # kick off all preproc tasks
        pre_futs = {pre_pool.submit(pre_worker, p): p for p in input_paths}
        post_futs: list = []

        # main loop – as soon as a preproc future is ready, push to GPU queue
        while pre_futs or not q_out.empty() or gpu_p.is_alive():
            # check finished preproc jobs
            done = [f for f in pre_futs if f.done()]
            for fut in done:
                q_in.put(fut.result())  # may block if queue full
                pre_futs.pop(fut)

            # drain anything the GPU has finished
            try:
                while True:
                    pack = q_out.get_nowait()
                    fut = post_pool.submit(post_worker, pack)
                    post_futs.append(fut)
                    processed += 1
                    if processed % 1 == 0:  # every 50 slices
                        console.print(f"{processed} done …", end="\r")
                    q_out.task_done()
            except queue.Empty:
                pass

            done_errs = [f for f in post_futs if f.done() and f.exception()]
            for f in done_errs:
                console.print(f"[red]{f.exception()}[/red]", highlight=False)
                post_futs.remove(f)

            time.sleep(0.01)

        # wait until GPU swallowed everything then poison-pill
        q_in.join()
        q_in.put(STOP)
        gpu_p.join()

        # wait for postproc tasks
        post_pool.shutdown(wait=True)

    dt = time.time() - t0
    console.print(f"[green]Done – processed {processed} slices in {dt:,.1f} s "
                  f"({processed / dt:,.2f} img/s)[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
