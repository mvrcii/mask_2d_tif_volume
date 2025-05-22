#!/usr/bin/env python3
"""
Minimal, transparent streaming pipeline:
  pre  (1 CPU proc)  →  q_in   →  gpu  (1 GPU proc)  →  q_out  →  post (1 CPU proc)

Every process logs to stderr; main process shows queue depths every second.
"""

import os, sys, time, queue, pathlib, cv2, tifffile, numpy as np, torch, signal
from multiprocessing import Process, JoinableQueue, Event, current_process, freeze_support
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from config import Config as C
from rich.console import Console
from skimage.io import imread, imsave
from skimage.transform import resize

console = Console()

# ----------------------------------------------------------------- helpers
def log(msg: str):
    p = current_process()
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}][{p.name}:{p.pid}] {msg}\n")
    sys.stderr.flush()


def pad_to_stride(img: np.ndarray, stride_xy: tuple[int, int]):
    h, w = img.shape
    sx, sy = stride_xy
    ph = (sx - h % sx) % sx
    pw = (sy - w % sy) % sy
    if ph or pw: img = np.pad(img, ((0, ph), (0, pw)), mode='edge')
    return img, (h, w)


# ---------- Stage A:  pre_proc: read + down-scale -------------------------------------
def pre_proc(q_in: JoinableQueue, stop: Event):
    try:
        for tif in sorted(pathlib.Path(C.input_dir).glob("*.tif")):
            img = imread(tif)
            original_dtype = img.dtype
            downscaled_shape = (
                img.shape[0] // C.downscale_factor,
                img.shape[1] // C.downscale_factor
            )
            img_ds = resize(
                img,
                downscaled_shape,
                preserve_range=True,
                anti_aliasing=True
            ).astype(original_dtype)

            q_in.put((tif.name, img_ds, img.shape, img.dtype.str))
            log(f"sent {tif.name}")
            if stop.is_set(): break
    except Exception as e:
        log(f"CRASH {e}")
        raise
    finally:
        q_in.put(None)  # sentinel
        log("finished")


def gpu_proc(q_in: JoinableQueue, q_out: JoinableQueue, stop: Event):
    try:
        device = torch.device('cuda')
        predictor = nnUNetPredictor(tile_step_size=C.tile_step_size,
                                    use_gaussian=True,
                                    use_mirroring=C.use_mirroring,
                                    perform_everything_on_device=True,
                                    device=device, verbose=False)

        ckpt = next(p for p in (pathlib.Path(C.model_dir) / "fold_0").glob("checkpoint_*.pth"))
        predictor.initialize_from_trained_model_folder(C.model_dir, use_folds=(0,), checkpoint_name=ckpt.name)
        predictor.network.to(device).half()

        pk = np.array(predictor.configuration_manager.pool_op_kernel_sizes)
        stride = tuple(int(np.prod(pk[:, i])) for i in range(2))
        log(f"stride {stride}")

        while True:
            item = q_in.get()
            if item is None: break
            fname, img_ds, orig_shape, dtype_str = item
            img_ds, crop_sz = pad_to_stride(img_ds, stride)

            img_ds = np.clip(img_ds, 0, 65000).astype(np.float32) / 65000.0
            tens = torch.from_numpy(img_ds).half().unsqueeze(0).unsqueeze(0).to(device)
            # tens = torch.from_numpy(img_ds).unsqueeze(0).unsqueeze(0).to(device).half()
            with torch.no_grad():
                logits = torch.sigmoid(predictor.network(tens))[0, 0].cpu().numpy()
            logits = logits[:crop_sz[0], :crop_sz[1]]
            q_out.put((fname, logits, orig_shape, dtype_str))
            log(f"done {fname}")
            if stop.is_set(): break
    except Exception as e:
        log(f"CRASH {e}")
        raise
    finally:
        q_out.put(None)
        log("finished")


def post_proc(q_out: JoinableQueue, stop: Event):
    try:
        out_dir = pathlib.Path(C.output_dir);
        out_dir.mkdir(exist_ok=True, parents=True)
        while True:
            item = q_out.get()
            if item is None: break
            fname, logits, orig_shape, dtype_str = item
            mask = (logits < C.threshold).astype(np.uint8)
            H, W = orig_shape
            mask_us = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            img = tifffile.imread(pathlib.Path(C.input_dir) / fname).astype(np.dtype(dtype_str))
            tifffile.imwrite(out_dir / fname, img * mask_us, compression="deflate")
            log(f"saved {fname}")
            if stop.is_set(): break
    except Exception as e:
        log(f"CRASH {e}")
        raise
    finally:
        log("finished")


# ----------------------------------------------------------------- main
def main():
    freeze_support()

    os.makedirs(C.output_dir, exist_ok=True)

    q_in = JoinableQueue(maxsize=C.queue_size)
    q_out = JoinableQueue(maxsize=C.queue_size)
    stop = Event()

    pre = Process(target=pre_proc, args=(q_in, stop), name="PRE")
    gpu = Process(target=gpu_proc, args=(q_in, q_out, stop), name="GPU")
    post = Process(target=post_proc, args=(q_out, stop), name="POST")

    for p in (pre, gpu, post): p.start()

    try:
        while any(p.is_alive() for p in (pre, gpu, post)):
            log(f"q_in={q_in.qsize()} q_out={q_out.qsize()}")
            time.sleep(1)
            for p in (pre, gpu, post):
                if not p.is_alive() and p.exitcode not in (0, None):
                    stop.set()
                    raise RuntimeError(f"{p.name} crashed with exitcode {p.exitcode}")
    except KeyboardInterrupt:
        stop.set()
    finally:
        for p in (pre, gpu, post): p.join()


if __name__ == "__main__":
    sys.exit(main())
