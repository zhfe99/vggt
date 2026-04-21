import argparse
from pathlib import Path

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def _read_image_list(txt_path: str) -> list[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    image_paths = [ln for ln in lines if ln and not ln.startswith("#")]
    return image_paths


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export per-image depth using VGGT")
    parser.add_argument("--image_list", type=str, required=True, help="Path to txt file, one image path per line")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
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

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _read_image_list(args.image_list)
    if len(image_paths) == 0:
        raise ValueError(f"No valid image paths found in {args.image_list}")

    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval()
    model = model.to(device)

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start : start + args.batch_size]

        images = load_and_preprocess_images(batch_paths, mode=args.preprocess_mode).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=dtype):
                preds = model(images)

        depth = preds["depth"]  # [B, S, H, W, 1] with B=1, S=N
        depth = depth.squeeze(0).detach().cpu().numpy()  # [S, H, W, 1]

        for i, img_path in enumerate(batch_paths):
            stem = Path(img_path).stem
            depth_i = depth[i, ..., 0]

            npy_path = out_dir / f"{stem}.depth.npy"
            np.save(npy_path, depth_i.astype(np.float32))

            if args.save_png:
                try:
                    import imageio.v3 as iio
                except Exception as e:
                    raise RuntimeError(
                        "--save_png requires imageio. Install e.g. `pip install imageio` or disable --save_png."
                    ) from e

                png_path = out_dir / f"{stem}.depth.png"
                iio.imwrite(png_path, _depth_to_uint16(depth_i))

        torch.cuda.empty_cache()

    print(f"Done. Exported {len(image_paths)} depth maps to: {out_dir}")


if __name__ == "__main__":
    main()
