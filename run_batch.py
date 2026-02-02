#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import hashlib
import html
from pathlib import Path

import numpy as np
from PIL import Image

from filters import mean_shift_filter_iter, median_shift_filter_iter

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                    paths.append(fp)
        else:
            g = glob.glob(inp)
            if g:
                paths.extend([Path(x) for x in sorted(g)])
            elif p.exists():
                paths.append(p)
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im).astype(np.float32) / 255.0  # H W 4
    rgb = arr[..., :3]
    a = arr[..., 3:4]
    return rgb, a


def save_rgb(path: Path, rgb: np.ndarray, *, alpha: np.ndarray | None = None) -> None:
    rgb8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    if alpha is not None:
        a8 = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
        out = np.concatenate([rgb8, a8], axis=-1)
        im = Image.fromarray(out)
    else:
        im = Image.fromarray(rgb8)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def resize_max(rgb: np.ndarray, alpha: np.ndarray, max_side: int | None) -> tuple[np.ndarray, np.ndarray]:
    if max_side is None:
        return rgb, alpha
    h, w = rgb.shape[:2]
    if max(h, w) <= max_side:
        return rgb, alpha
    scale = max_side / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    im_rgba = np.concatenate([rgb, alpha], axis=-1)
    pil = Image.fromarray(np.clip(np.round(im_rgba * 255.0), 0, 255).astype(np.uint8))
    pil = pil.resize((nw, nh), Image.BICUBIC)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return arr[..., :3], arr[..., 3:4]


def _safe_stem(path: Path) -> str:
    stem = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in path.stem).strip("._")
    stem = stem or "img"
    h = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{h}"


def _iter_dir(name: str, t: int) -> str:
    return f"{name}/iter_{t:03d}"


def write_html(outdir: Path, rows: list[dict], params: dict) -> None:
    max_iter = int(params["max_iter"])
    esc = html.escape
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>MeanShiftFilter vs MedianShiftFilter (Iterations)</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
         margin: 24px; color:#111; }}
  .meta {{ color:#444; line-height:1.4; margin-bottom: 16px; }}
  .controls {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin: 14px 0 18px; }}
  .controls input[type=range] {{ width: min(520px, 90vw); }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
  .card {{ border: 1px solid #e6e6e6; border-radius: 10px; padding: 14px; }}
  .row {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; align-items:start; }}
  figure {{ margin:0; }}
  figcaption {{ font-size: 12px; color:#555; margin-top:6px; word-break: break-all; }}
  img {{ width:100%; height:auto; border-radius: 8px; border:1px solid #f0f0f0; background:#fafafa; }}
  .fn {{ font-weight: 600; margin-bottom: 10px; }}
  button {{ padding: 6px 10px; border-radius: 8px; border:1px solid #ddd; background:#fff; cursor:pointer; }}
  button:hover {{ background:#f7f7f7; }}
  code {{ background:#f7f7f7; padding: 2px 4px; border-radius: 6px; }}
  @media (max-width: 900px) {{
    .row {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<h1>MeanShiftFilter vs MedianShiftFilter</h1>
<div class="meta">
  <div><b>Params</b>: r={params['r']} d={params['d']} max_iter={params['max_iter']} distance={esc(str(params['distance']))} median_mode={esc(str(params['median_mode']))}</div>
  <div><b>Compute</b>: requested backend={esc(str(params['requested_backend']))} device={esc(str(params['requested_device']))} · used backend={esc(str(params['backend']))} device={esc(str(params['device']))}</div>
  <div>Tip: geometric median mode can be slow. Use <code>--resize_max</code>, small <code>--r</code>, and small <code>--max_iter</code>.</div>
</div>

<div class="controls">
  <button id="playBtn" type="button">Play</button>
  <div><b>Iteration</b>: <span id="iterLabel">0/{max_iter}</span></div>
  <input id="iterSlider" type="range" min="0" max="{max_iter}" value="0" step="1" />
</div>

<div class="grid">
"""

    for row in rows:
        disp = esc(row["display_name"])
        safe = esc(row["safe"])
        orig = esc(row["orig"])
        ms_pattern = esc(row["ms_pattern"])
        med_pattern = esc(row["med_pattern"])
        html_text += f"""
  <div class="card">
    <div class="fn">{disp}</div>
    <div class="row">
      <figure>
        <img src="{orig}" alt="original">
        <figcaption>Original</figcaption>
      </figure>
      <figure>
        <img class="iter-img" data-original="{orig}" data-pattern="{ms_pattern}" src="{orig}" alt="meanshift">
        <figcaption>MeanShiftFilter</figcaption>
      </figure>
      <figure>
        <img class="iter-img" data-original="{orig}" data-pattern="{med_pattern}" src="{orig}" alt="medianshift">
        <figcaption>MedianShiftFilter ({esc(str(params['median_mode']))})</figcaption>
      </figure>
    </div>
  </div>
"""

    html_text += f"""
</div>

<script>
(() => {{
  const maxIter = {max_iter};
  const slider = document.getElementById('iterSlider');
  const label = document.getElementById('iterLabel');
  const btn = document.getElementById('playBtn');
  const imgs = Array.from(document.querySelectorAll('img.iter-img'));
  let timer = null;

  function pad3(n) {{
    return String(n).padStart(3, '0');
  }}

  function setIter(iter) {{
    label.textContent = `${{iter}}/${{maxIter}}`;
    imgs.forEach(img => {{
      if (iter === 0) {{
        img.src = img.dataset.original;
        return;
      }}
      img.src = img.dataset.pattern.replace('{{iter}}', pad3(iter));
    }});
  }}

  function stop() {{
    if (timer) {{
      clearInterval(timer);
      timer = null;
    }}
    btn.textContent = 'Play';
  }}

  function play() {{
    if (timer) return;
    btn.textContent = 'Pause';
    timer = setInterval(() => {{
      const next = (Number(slider.value) + 1) % (maxIter + 1);
      slider.value = String(next);
      setIter(next);
    }}, 700);
  }}

  slider.addEventListener('input', (e) => {{
    stop();
    setIter(Number(e.target.value));
  }});

  btn.addEventListener('click', () => {{
    if (timer) stop();
    else play();
  }});

  setIter(0);
}})();
</script>
</body>
</html>
"""

    (outdir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Directory, glob(s), or file paths")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--d", type=float, default=0.10)
    ap.add_argument("--max_iter", type=int, default=1)
    ap.add_argument("--distance", type=str, default="euclidean", choices=["euclidean", "sqeuclidean", "manhattan", "cosine"])
    ap.add_argument("--median_mode", type=str, default="geometric", choices=["geometric", "component"])
    ap.add_argument("--gm_eps", type=float, default=1e-5)
    ap.add_argument("--gm_max_iter", type=int, default=50)
    ap.add_argument("--resize_max", type=int, default=None)
    ap.add_argument("--backend", type=str, default="auto", choices=["auto", "numpy", "torch"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--chunk_pixels", type=int, default=16384)
    args = ap.parse_args()

    resolved_backend = args.backend
    resolved_device = args.device
    if resolved_backend == "auto":
        try:
            import torch  # noqa: F401

            resolved_backend = "torch"
        except Exception:
            resolved_backend = "numpy"
    if resolved_device == "auto":
        if resolved_backend == "torch":
            try:
                import torch

                resolved_device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
            except Exception:
                resolved_device = "cpu"
        else:
            resolved_device = "cpu"

    paths = list_images(args.inputs)
    if not paths:
        raise SystemExit("No images found.")

    outdir = Path(args.outdir)
    (outdir / "original").mkdir(parents=True, exist_ok=True)
    (outdir / "meanshift").mkdir(parents=True, exist_ok=True)
    (outdir / "medianshift").mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm

        it = tqdm(paths, desc="Processing", unit="img")
    except Exception:
        it = paths

    rows: list[dict] = []
    for p in it:
        rgb, a = load_image(p)
        rgb, a = resize_max(rgb, a, args.resize_max)

        safe = _safe_stem(p)
        orig_path = outdir / "original" / f"{safe}.png"
        save_rgb(orig_path, rgb, alpha=a)

        for t, ms in enumerate(
            mean_shift_filter_iter(
                rgb,
                args.r,
                args.d,
                max_iter=args.max_iter,
                distance=args.distance,
                backend=resolved_backend,
                device=resolved_device,
                chunk_pixels=args.chunk_pixels,
            ),
            start=1,
        ):
            ms_path = outdir / _iter_dir("meanshift", t) / f"{safe}.png"
            save_rgb(ms_path, ms, alpha=a)

        for t, med in enumerate(
            median_shift_filter_iter(
                rgb,
                args.r,
                args.d,
                max_iter=args.max_iter,
                distance=args.distance,
                median_mode=args.median_mode,
                gm_eps=args.gm_eps,
                gm_max_iter=args.gm_max_iter,
                backend=resolved_backend,
                device=resolved_device,
                chunk_pixels=args.chunk_pixels,
            ),
            start=1,
        ):
            med_path = outdir / _iter_dir("medianshift", t) / f"{safe}.png"
            save_rgb(med_path, med, alpha=a)

        rows.append(
            {
                "display_name": p.name,
                "safe": safe,
                "orig": str(orig_path.relative_to(outdir)).replace("\\", "/"),
                "ms_pattern": (f"{_iter_dir('meanshift', 0)}/{safe}.png").replace("iter_000", "iter_{iter}"),
                "med_pattern": (f"{_iter_dir('medianshift', 0)}/{safe}.png").replace("iter_000", "iter_{iter}"),
            }
        )

    params = {
        "r": args.r,
        "d": args.d,
        "max_iter": args.max_iter,
        "distance": args.distance,
        "median_mode": args.median_mode,
        "requested_backend": args.backend,
        "requested_device": args.device,
        "backend": resolved_backend,
        "device": resolved_device,
    }
    write_html(outdir, rows, params)
    print(f"Done. Open: {outdir / 'index.html'}")


if __name__ == "__main__":
    main()
