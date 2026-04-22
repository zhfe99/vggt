import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def _read_image_list(txt_path: str) -> list[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    image_paths = [ln for ln in lines if ln and not ln.startswith("#")]
    return image_paths


def _read_image_dir(image_dir: str, *, recursive: bool) -> list[str]:
    p = Path(image_dir)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"--image_dir must be an existing directory, got: {image_dir}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    it = p.rglob("*") if recursive else p.iterdir()
    files = [f for f in it if f.is_file() and f.suffix.lower() in exts]
    files.sort()
    return [str(f) for f in files]


def _depth_to_uint16(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    finite = np.isfinite(depth)
    if not np.any(finite):
        return np.zeros(depth.shape, dtype=np.uint16)

    vmin = float(np.percentile(depth[finite], 1.0))
    vmax = float(np.percentile(depth[finite], 99.0))
    if vmax <= vmin:
        return np.zeros(depth.shape, dtype=np.uint16)

    depth_norm = (depth - vmin) / (vmax - vmin)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    return (depth_norm * 65535.0).round().astype(np.uint16)


def _depth_u16_to_u8(depth_u16: np.ndarray) -> np.ndarray:
    return (depth_u16.astype(np.float32) / 257.0).round().clip(0, 255).astype(np.uint8)


def _project_depth_to_original(
    *,
    depth_u8: np.ndarray,
    orig_w: int,
    orig_h: int,
    preprocess_mode: str,
    target_size: int = 518,
) -> np.ndarray | None:
    if depth_u8.ndim != 2:
        return None

    if preprocess_mode not in {"pad", "crop"}:
        return None

    if preprocess_mode == "pad":
        if orig_w >= orig_h:
            new_w = target_size
            new_h = int(round(orig_h * (new_w / orig_w) / 14.0) * 14)
        else:
            new_h = target_size
            new_w = int(round(orig_w * (new_h / orig_h) / 14.0) * 14)

        pad_top = (target_size - new_h) // 2
        pad_left = (target_size - new_w) // 2
        crop = depth_u8[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
        crop_img = Image.fromarray(crop, mode="L")
        out = crop_img.resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
        return np.asarray(out)

    new_w = target_size
    new_h = int(round(orig_h * (new_w / orig_w) / 14.0) * 14)
    if new_h <= 0:
        return None
    if new_h > target_size:
        start_y = (new_h - target_size) // 2
        full = np.zeros((new_h, new_w), dtype=np.uint8)
        full[start_y : start_y + target_size, :] = depth_u8
        full_img = Image.fromarray(full, mode="L")
    else:
        full_img = Image.fromarray(depth_u8, mode="L")

    out = full_img.resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
    return np.asarray(out)


def _safe_load_images(
    image_paths: list[str],
    *,
    mode: str,
    device: str,
) -> tuple[torch.Tensor | None, list[str]]:
    try:
        images = load_and_preprocess_images(image_paths, mode=mode).to(device)
        return images, image_paths
    except Exception as e:
        print(f"Warning: failed to load a batch of {len(image_paths)} images, will try per-image. Error: {e}")

    ok_images: list[torch.Tensor] = []
    ok_paths: list[str] = []
    for p in image_paths:
        try:
            img = load_and_preprocess_images([p], mode=mode).to(device)
        except Exception as e:
            print(f"Warning: skipping unreadable image: {p}. Error: {e}")
            continue
        ok_images.append(img)
        ok_paths.append(p)

    if len(ok_images) == 0:
        return None, []

    images = torch.cat(ok_images, dim=0)
    return images, ok_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-image depth using VGGT")
    parser.add_argument("--image_list", type=str, default=None, help="Path to txt file, one image path per line")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to a directory of images")
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        default=False,
        help="Disable recursive scan when using --image_dir (recursive is enabled by default)",
    )
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="pad",
        choices=["pad", "crop"],
        help="Preprocess mode passed to load_and_preprocess_images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu. Default: auto-detect (prefer cuda)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of images per forward pass (reduce if you OOM)",
    )
    parser.add_argument(
        "--save_png",
        action="store_true",
        default=False,
        help="Also save a 16-bit PNG for visualization (normalized per-image)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if (args.image_list is None) == (args.image_dir is None):
        raise ValueError("Specify exactly one of --image_list or --image_dir")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.out_dir is None:
        if args.image_dir is None:
            raise ValueError("--out_dir is required when using --image_list")
        out_dir = Path(f"{args.image_dir}_vggt")
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_root: Path | None = None
    if args.image_list is not None:
        image_paths = _read_image_list(args.image_list)
    else:
        image_root = Path(args.image_dir)
        image_paths = _read_image_dir(args.image_dir, recursive=(not args.no_recursive))
    if len(image_paths) == 0:
        src = args.image_list if args.image_list is not None else args.image_dir
        raise ValueError(f"No valid image paths found in {src}")

    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval()
    model = model.to(device)

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]

        images, ok_paths = _safe_load_images(batch_paths, mode=args.preprocess_mode, device=device)
        if images is None or len(ok_paths) == 0:
            continue

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype):
                try:
                    preds = model(images)
                except Exception as e:
                    print(f"Warning: model forward failed for a batch (size={len(ok_paths)}), skipping. Error: {e}")
                    continue

        depth = preds["depth"]  # [B, S, H, W, 1] with B=1, S=N
        depth = depth.squeeze(0).detach().cpu().numpy()  # [S, H, W, 1]

        for i, img_path in enumerate(ok_paths):
            if image_root is not None:
                rel = Path(img_path).relative_to(image_root)
                stem = rel.stem
                img_out_dir = out_dir / rel.parent
                img_out_dir.mkdir(parents=True, exist_ok=True)
            else:
                stem = Path(img_path).stem
                img_out_dir = out_dir
            depth_i = depth[i, ..., 0]

            npy_path = img_out_dir / f"{stem}.depth.npy"
            np.save(npy_path, depth_i.astype(np.float32))

            if args.save_png:
                try:
                    import imageio.v3 as iio
                except Exception as e:
                    raise RuntimeError(
                        "--save_png requires imageio. Install e.g. `pip install imageio` or disable --save_png."
                    ) from e

                png_path = img_out_dir / f"{stem}.depth.png"
                depth_u16 = _depth_to_uint16(depth_i)
                iio.imwrite(png_path, depth_u16)

                try:
                    img_in = images[i].detach().float().cpu().numpy()  # [3, H, W] in [0,1]
                    img_in = np.transpose(img_in, (1, 2, 0))
                    img_u8 = (np.clip(img_in, 0.0, 1.0) * 255.0).round().astype(np.uint8)

                    depth_u8 = _depth_u16_to_u8(depth_u16)
                    depth_rgb = np.repeat(depth_u8[..., None], 3, axis=2)

                    if img_u8.shape[:2] == depth_rgb.shape[:2]:
                        debug = np.concatenate([img_u8, depth_rgb], axis=1)
                        debug_path = img_out_dir / f"{stem}.debug.png"
                        iio.imwrite(debug_path, debug)

                        alpha = 0.65
                        overlay = (
                            (1.0 - alpha) * img_u8.astype(np.float32) + alpha * depth_rgb.astype(np.float32)
                        ).round().clip(0, 255).astype(np.uint8)

                        gx = np.abs(np.diff(depth_u8.astype(np.int16), axis=1, prepend=depth_u8[:, :1]))
                        gy = np.abs(np.diff(depth_u8.astype(np.int16), axis=0, prepend=depth_u8[:1, :]))
                        edges = (gx + gy) > 20
                        overlay[edges] = np.array([255, 0, 0], dtype=np.uint8)
                        overlay_path = img_out_dir / f"{stem}.overlay.png"
                        iio.imwrite(overlay_path, overlay)

                    try:
                        orig_img = Image.open(img_path)
                        if orig_img.mode == "RGBA":
                            background = Image.new("RGBA", orig_img.size, (255, 255, 255, 255))
                            orig_img = Image.alpha_composite(background, orig_img)
                        orig_img = orig_img.convert("RGB")
                        orig_w, orig_h = orig_img.size
                        depth_on_orig = _project_depth_to_original(
                            depth_u8=depth_u8,
                            orig_w=orig_w,
                            orig_h=orig_h,
                            preprocess_mode=args.preprocess_mode,
                        )
                        if depth_on_orig is not None:
                            orig_u8 = np.asarray(orig_img, dtype=np.uint8)
                            depth_on_orig_rgb = np.repeat(depth_on_orig[..., None], 3, axis=2)
                            alpha = 0.65
                            orig_overlay = (
                                (1.0 - alpha) * orig_u8.astype(np.float32)
                                + alpha * depth_on_orig_rgb.astype(np.float32)
                            ).round().clip(0, 255).astype(np.uint8)

                            gx = np.abs(
                                np.diff(depth_on_orig.astype(np.int16), axis=1, prepend=depth_on_orig[:, :1])
                            )
                            gy = np.abs(
                                np.diff(depth_on_orig.astype(np.int16), axis=0, prepend=depth_on_orig[:1, :])
                            )
                            edges = (gx + gy) > 20
                            orig_overlay[edges] = np.array([255, 0, 0], dtype=np.uint8)
                            orig_overlay_path = img_out_dir / f"{stem}.orig_overlay.png"
                            iio.imwrite(orig_overlay_path, orig_overlay)
                    except Exception as e:
                        print(f"Warning: failed to write original-space overlay for {img_path}. Error: {e}")
                except Exception as e:
                    print(f"Warning: failed to write debug composite for {img_path}. Error: {e}")

        torch.cuda.empty_cache()

    print(f"Done. Exported {len(image_paths)} depth maps to: {out_dir}")


if __name__ == "__main__":
    main()
