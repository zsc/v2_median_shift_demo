下面先把 **Mathematica / Wolfram Language 的 `MeanShiftFilter`** 讲清楚（按官方语义），再给出你要的 **median / 几何中位数版**（把 mean-shift 更新里的 mean 换成 median / geometric median），最后提供一份可以直接丢给 **codex / gemini-cli** 的 **SPEC Markdown**（含 Python+HTML 的实现要求与验收标准），并附一个**可运行的参考实现（单文件 Python，会批量读图并生成 HTML 展示页）**。

---

## 1) Mathematica `MeanShiftFilter` 到底做了什么？

Wolfram 文档里定义的是：

* `MeanShiftFilter[data, r, d]`：**对每个位置**，只在一个“半径为 r 的邻域（range-r neighborhood）”里看数据点；然后只把**值与中心点相差不超过 d**（按 `DistanceFunction`）的那些邻域点拿来做更新，用它们的**均值**替换当前位置。([Wolfram 参考文献][1])
* `MeanShiftFilter[data, {r1,r2,...}, d]`：各维用不同半径。([Wolfram 参考文献][1])
* 多通道（彩色图）时，“距离”是在**通道向量**之间算（例如 RGB 向量）。([Wolfram 参考文献][1])
* 边界处使用**更小的邻域**（不是强行 padding 复制外部像素）。([Wolfram 参考文献][1])
* 选项：

  * `DistanceFunction`（默认欧氏距离）([Wolfram 参考文献][1])
  * `MaxIterations`（默认 1 次）([Wolfram 参考文献][1])

同时文档明确：它在每个局部邻域上应用的函数是 `MeanShift`。([Wolfram 参考文献][1])
而 `MeanShift[list, d]` 的描述是：“把每个元素替换为与它相差 < d 的元素的均值”，并支持 `MaxIterations / Tolerance / Weights` 等选项。([Wolfram 参考文献][2])

### 用“图像版”伪公式重述（贴近 Wolfram 的描述）

令当前迭代图像为 (I^{(t)})，像素位置 (p)，像素值（向量）(x^{(t)} = I^{(t)}(p))。

* 空间邻域（2D）：
  [
  \mathcal N_r(p)={q\mid |q-p|_\infty\le r}
  ]
  （一个 ((2r+1)\times(2r+1)) 的方窗；边界处窗会变小）

* 候选集合（值域阈值）：
  [
  S={I^{(t)}(q)\mid q\in \mathcal N_r(p),\ \text{dist}(I^{(t)}(q), x^{(t)})\le d}
  ]

* 更新（MeanShiftFilter 的“mean”）：
  [
  I^{(t+1)}(p)=\text{mean}(S)
  ]

* 做 `MaxIterations` 次（默认 1 次）。([Wolfram 参考文献][1])

> 这和经典论文里“在联合的 spatial-range 空间里做动态窗口 mean shift 直到收敛”的完整版 mean-shift filtering很像，但 Wolfram 的 `MeanShiftFilter` 文档表述更偏向“局部窗口 + 值域阈值 + 迭代”。经典 mean shift 的“联合空间/收敛到 mode/输出收敛点的 range 分量”等背景可参考 Comaniciu & Meer 的论文。([Comaniciu][3])

---

## 2) 你要的 median 版本：把 mean 换成 median / 几何中位数

你说的“把 mean-shift 更新里的 mean 替换成 median/几何中位数”，最自然的定义是：

* **MedianShiftFilter（标量）**：更新用 **median(S)**。
* **MedianShiftFilter（向量/彩色）**：median 的多维推广有两种常用选择

  1. **逐通道 median（coordinate-wise median）**：对 RGB 各通道分别取 median（快，但不是严格的“多元中位数”）。
  2. **几何中位数 geometric median（L1-median / spatial median）**：
     [
     \operatorname*{argmin}*{y}\sum*{s\in S}|y-s|_2
     ]
     常用 **Weiszfeld 算法**迭代求解。([维基百科][4])

你提到“median/几何中位数”，我建议默认策略是：

* 灰度：median
* 彩色：**默认 geometric median**（更“正宗”），但提供 `--median_mode component` 作为加速备选。

---

## 3) 可直接交给 codex/gemini-cli 的 SPEC（Markdown）

> 下面这个代码块就是你要的“SPEC markdown”（方便复制粘贴给 codex / gemini-cli）。**注意**：为了便于复制执行，我把实现细节、CLI、文件结构、验收标准都写进去了。

```markdown
# SPEC: Mathematica-like MeanShiftFilter + Median(Geometric) Variant (Python + HTML batch demo)

## Goal
Implement:
1) MeanShiftFilter (Mathematica/Wolfram-like semantics)
2) MedianShiftFilter (replace mean update with median; for vectors use geometric median by default)

Provide:
- Python CLI that can process a *set of images* (directory or multiple paths/globs)
- Outputs filtered images + an `index.html` gallery comparing Original vs MeanShift vs MedianShift
- Clean, readable code suitable for extension.

## References (behavioral targets)
- MeanShiftFilter[data, r, d]: for each sample, look within spatial radius r and average only values within distance d of the sample value; supports MaxIterations and DistanceFunction; boundary uses smaller neighborhoods.
- Multi-channel distance computed between channel vectors.

## Deliverables / Repo layout
- `filters.py`
  - `mean_shift_filter(img, r, d, *, max_iter=1, distance="euclidean") -> np.ndarray`
  - `median_shift_filter(img, r, d, *, max_iter=1, distance="euclidean", median_mode="geometric",
                        gm_eps=1e-5, gm_max_iter=50) -> np.ndarray`
- `run_batch.py`
  - CLI: load multiple images, run both filters, save outputs, generate HTML
- `templates/index.template.html` (optional; can be generated in code)
- `requirements.txt`

## CLI
`python run_batch.py --inputs <dir_or_glob_or_paths...> --outdir out \
  --r 4 --d 0.10 --max_iter 1 --distance euclidean \
  --median_mode geometric --resize_max 800`

Arguments:
- `--inputs`: one or more paths/globs; if a directory, include common image extensions (png/jpg/jpeg/webp)
- `--outdir`: output directory; write:
  - `original/<name>.png` (copy or re-encode)
  - `meanshift/<name>.png`
  - `medianshift/<name>.png`
  - `index.html`
- `--r`: integer radius (pixels). Neighborhood is a square window (2r+1)x(2r+1).
- `--d`: float threshold in *value space distance* (on normalized float image [0,1]).
- `--max_iter`: number of outer iterations (default 1).
- `--distance`: `euclidean|manhattan|sqeuclidean|cosine`
- `--median_mode`: `geometric|component`
- `--resize_max`: optional int; if set, resize so max(H,W) <= resize_max to avoid very slow runs.

Exit code:
- 0 on success; non-zero on failures.

## Core algorithm (Mathematica-like MeanShiftFilter)
Given current image I^t, for each pixel p:
1) Take spatial neighborhood N_r(p) (crop at boundaries => smaller neighborhoods)
2) Select S = { I^t(q) in N_r(p) where dist(I^t(q), I^t(p)) <= d }
3) Update:
   - MeanShiftFilter: I^{t+1}(p) = mean(S)
   - MedianShiftFilter:
       - scalar: median(S)
       - vector (C>1):
           - component: per-channel median(S)
           - geometric: y = argmin_y sum_{s in S} ||y - s||_2  (Weiszfeld)
4) Repeat for `max_iter` iterations.

Important:
- Ensure the center pixel is always included (distance 0) so S is never empty.
- Boundary handling MUST mimic “smaller neighborhoods”, not reflection padding.

## Geometric median (Weiszfeld)
Input: points X (n x C), weights w (optional; default uniform)
Initialize y0 = mean(X).
Iterate up to gm_max_iter:
  di = ||Xi - y||
  If any di < gm_eps: return Xi (or y)
  y_next = sum( w_i * Xi / di ) / sum( w_i / di )
  Stop when ||y_next - y|| < gm_eps

## HTML gallery
Generate `index.html`:
- For each input image, show a row (or card) with:
  - Original
  - MeanShiftFilter result
  - MedianShiftFilter result
- Responsive grid layout (CSS).
- Display filename, and parameters used (r, d, max_iter, distance, median_mode).

No external dependencies required for HTML (pure static).

## Performance constraints / safety
- Warn user if geometric median mode on large images is slow; recommend `--resize_max` and small r.
- Implement images as float32 [0,1]; write PNG outputs (uint8).
- Handle grayscale and RGB; preserve alpha by filtering RGB and reattaching alpha (optional but nice).

## Acceptance tests (must pass)
1) Constant image unchanged for any r,d,max_iter.
2) If d == 0: output equals input (since only identical values selected; center always included).
3) If d is very large: MeanShiftFilter ~= MeanFilter with radius r (on same boundary rule).
4) Batch run produces:
   - output images in expected folders
   - index.html opens and shows all images.

## Dependencies
Minimum:
- numpy
- pillow

Optional (not required):
- tqdm (progress bar)
```

---

## 4) 单文件参考实现（Python：批量读图 + 两种滤波 + 生成 HTML）

> 这份是“能跑起来”的最小实现：
>
> * MeanShiftFilter：用**向量化滑窗**实现（速度还可以，但内存随 `r` 和图像变大）
> * MedianShiftFilter：
>
>   * `component`：向量化（快）
>   * `geometric`：逐像素 Weiszfeld（慢，建议小图 + 小 r + 少迭代）

把下面保存成 `run_batch.py`，然后 `python run_batch.py --inputs ./imgs --outdir out --r 4 --d 0.1 --median_mode geometric --resize_max 600`。

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------
# IO helpers
# ---------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(inputs):
    paths = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for ext in IMG_EXTS:
                paths.extend(sorted(p.rglob(f"*{ext}")))
        else:
            # glob or file
            g = glob.glob(inp)
            if g:
                paths.extend([Path(x) for x in sorted(g)])
            elif p.exists():
                paths.append(p)
    # de-dup keep order
    seen = set()
    uniq = []
    for p in paths:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    return uniq


def load_image(path):
    im = Image.open(path).convert("RGBA")
    arr = np.asarray(im).astype(np.float32) / 255.0  # H W 4
    rgb = arr[..., :3]
    a = arr[..., 3:4]
    return rgb, a


def save_rgb(path, rgb, alpha=None):
    rgb8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    if alpha is not None:
        a8 = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
        out = np.concatenate([rgb8, a8], axis=-1)
        im = Image.fromarray(out, mode="RGBA")
    else:
        im = Image.fromarray(rgb8, mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def resize_max(rgb, alpha, max_side):
    if max_side is None:
        return rgb, alpha
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return rgb, alpha
    scale = max_side / float(m)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    im_rgba = np.concatenate([rgb, alpha], axis=-1)
    pil = Image.fromarray(np.clip(np.round(im_rgba * 255.0), 0, 255).astype(np.uint8), mode="RGBA")
    pil = pil.resize((nw, nh), Image.BICUBIC)
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return arr[..., :3], arr[..., 3:4]


# ---------------------------
# Distance functions
# ---------------------------

def compute_dist(window, center, kind):
    # window: (..., C), center: (..., C) broadcastable
    diff = window - center
    if kind == "euclidean":
        return np.sqrt(np.nansum(diff * diff, axis=-1))
    if kind == "sqeuclidean":
        return np.nansum(diff * diff, axis=-1)
    if kind == "manhattan":
        return np.nansum(np.abs(diff), axis=-1)
    if kind == "cosine":
        # 1 - cos similarity between vectors
        a = window
        b = center
        num = np.nansum(a * b, axis=-1)
        da = np.sqrt(np.nansum(a * a, axis=-1))
        db = np.sqrt(np.nansum(b * b, axis=-1))
        return 1.0 - (num / (da * db + 1e-12))
    raise ValueError(f"Unknown distance: {kind}")


# ---------------------------
# MeanShiftFilter (vectorized, boundary = smaller neighborhood via NaN pad)
# ---------------------------

def mean_shift_filter(img, r, d, *, max_iter=1, distance="euclidean"):
    """
    img: H W C float32 in [0,1]
    """
    from numpy.lib.stride_tricks import sliding_window_view

    assert img.ndim == 3
    h, w, c = img.shape
    win = 2 * r + 1

    cur = img.astype(np.float32, copy=True)

    for _ in range(max_iter):
        # NaN pad => boundary neighborhoods effectively smaller (NaNs excluded)
        pad = ((r, r), (r, r), (0, 0))
        padded = np.pad(cur, pad_width=pad, mode="constant", constant_values=np.nan)

        # windows: H W win win C
        windows = sliding_window_view(padded, (win, win), axis=(0, 1))
        # sliding_window_view keeps channel axis intact at the end
        # windows shape: (H, W, win, win, C)

        center = cur[:, :, None, None, :]  # H W 1 1 C

        valid = ~np.isnan(windows[..., 0])  # H W win win
        dist = compute_dist(windows, center, distance)
        mask = valid & (dist <= d)

        # Ensure center included (should already be dist==0 unless NaN)
        mask[:, :, r, r] = True

        denom = np.sum(mask, axis=(2, 3)).astype(np.float32)  # H W
        numer = np.nansum(windows * mask[..., None], axis=(2, 3))  # H W C
        cur = numer / denom[..., None]

    return cur


# ---------------------------
# MedianShiftFilter (component-wise fast version)
# ---------------------------

def median_shift_filter_component(img, r, d, *, max_iter=1, distance="euclidean"):
    """
    Component-wise median per channel, using NaN mask + nanmedian.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    h, w, c = img.shape
    win = 2 * r + 1
    cur = img.astype(np.float32, copy=True)

    for _ in range(max_iter):
        pad = ((r, r), (r, r), (0, 0))
        padded = np.pad(cur, pad_width=pad, mode="constant", constant_values=np.nan)
        windows = sliding_window_view(padded, (win, win), axis=(0, 1))  # H W win win C
        center = cur[:, :, None, None, :]

        valid = ~np.isnan(windows[..., 0])
        dist = compute_dist(windows, center, distance)
        mask = valid & (dist <= d)
        mask[:, :, r, r] = True

        masked = np.where(mask[..., None], windows, np.nan)
        cur = np.nanmedian(masked, axis=(2, 3))

    return cur


# ---------------------------
# Geometric median (Weiszfeld)
# ---------------------------

def geometric_median_weiszfeld(X, eps=1e-5, max_iter=50):
    """
    X: (n, C) float
    returns y: (C,)
    """
    if X.shape[0] == 1:
        return X[0]
    y = np.nanmean(X, axis=0)
    for _ in range(max_iter):
        diff = X - y[None, :]
        dist = np.linalg.norm(diff, axis=1)
        # if y hits a point
        min_i = np.argmin(dist)
        if dist[min_i] < eps:
            return X[min_i]
        w = 1.0 / (dist + 1e-12)
        y_next = (X * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_next - y) < eps:
            return y_next
        y = y_next
    return y


def median_shift_filter_geometric(img, r, d, *, max_iter=1, distance="euclidean", gm_eps=1e-5, gm_max_iter=50):
    """
    Slow reference: per-pixel neighborhood gather + Weiszfeld.
    """
    h, w, c = img.shape
    cur = img.astype(np.float32, copy=True)

    # Pre-pad with NaN to implement smaller boundary neighborhoods
    for _ in range(max_iter):
        padded = np.pad(cur, ((r, r), (r, r), (0, 0)), mode="constant", constant_values=np.nan)
        out = np.empty_like(cur)

        for i in range(h):
            for j in range(w):
                patch = padded[i:i + 2 * r + 1, j:j + 2 * r + 1, :]  # win win C
                center = cur[i, j, :][None, None, :]

                valid = ~np.isnan(patch[..., 0])
                dist = compute_dist(patch, center, distance)  # win win
                mask = valid & (dist <= d)
                mask[r, r] = True

                pts = patch[mask]  # (n, C)
                if pts.shape[0] == 0:
                    out[i, j, :] = cur[i, j, :]
                else:
                    out[i, j, :] = geometric_median_weiszfeld(pts, eps=gm_eps, max_iter=gm_max_iter)

        cur = out

    return cur


def median_shift_filter(img, r, d, *, max_iter=1, distance="euclidean", median_mode="geometric", gm_eps=1e-5, gm_max_iter=50):
    if median_mode == "component":
        return median_shift_filter_component(img, r, d, max_iter=max_iter, distance=distance)
    if median_mode == "geometric":
        return median_shift_filter_geometric(img, r, d, max_iter=max_iter, distance=distance, gm_eps=gm_eps, gm_max_iter=gm_max_iter)
    raise ValueError(f"Unknown median_mode: {median_mode}")


# ---------------------------
# HTML gallery
# ---------------------------

def write_html(outdir, rows, params):
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>MeanShiftFilter vs MedianShiftFilter</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
         margin: 24px; color:#111; }}
  .meta {{ color:#444; line-height:1.4; margin-bottom: 16px; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
  .card {{ border: 1px solid #e6e6e6; border-radius: 10px; padding: 14px; }}
  .row {{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; align-items:start; }}
  figure {{ margin:0; }}
  figcaption {{ font-size: 12px; color:#555; margin-top:6px; word-break: break-all; }}
  img {{ width:100%; height:auto; border-radius: 8px; border:1px solid #f0f0f0; background:#fafafa; }}
  .fn {{ font-weight: 600; margin-bottom: 10px; }}
  @media (max-width: 900px) {{
    .row {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<h1>MeanShiftFilter vs MedianShiftFilter</h1>
<div class="meta">
  <div><b>Params</b>: r={params['r']} d={params['d']} max_iter={params['max_iter']} distance={params['distance']} median_mode={params['median_mode']}</div>
  <div>Tip: geometric median mode can be slow on large images. Use --resize_max and small r.</div>
</div>

<div class="grid">
"""
    for fn, orig, ms, med in rows:
        html += f"""
  <div class="card">
    <div class="fn">{fn}</div>
    <div class="row">
      <figure>
        <img src="{orig}" alt="original">
        <figcaption>Original</figcaption>
      </figure>
      <figure>
        <img src="{ms}" alt="meanshift">
        <figcaption>MeanShiftFilter</figcaption>
      </figure>
      <figure>
        <img src="{med}" alt="medianshift">
        <figcaption>MedianShiftFilter ({params['median_mode']})</figcaption>
      </figure>
    </div>
  </div>
"""
    html += """
</div>
</body>
</html>
"""
    out_path = Path(outdir) / "index.html"
    out_path.write_text(html, encoding="utf-8")


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Directory, glob(s), or file paths")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--d", type=float, default=0.10)
    ap.add_argument("--max_iter", type=int, default=1)
    ap.add_argument("--distance", type=str, default="euclidean",
                    choices=["euclidean", "sqeuclidean", "manhattan", "cosine"])
    ap.add_argument("--median_mode", type=str, default="geometric", choices=["geometric", "component"])
    ap.add_argument("--gm_eps", type=float, default=1e-5)
    ap.add_argument("--gm_max_iter", type=int, default=50)
    ap.add_argument("--resize_max", type=int, default=None)
    args = ap.parse_args()

    paths = list_images(args.inputs)
    if not paths:
        raise SystemExit("No images found.")

    outdir = Path(args.outdir)
    (outdir / "original").mkdir(parents=True, exist_ok=True)
    (outdir / "meanshift").mkdir(parents=True, exist_ok=True)
    (outdir / "medianshift").mkdir(parents=True, exist_ok=True)

    rows = []
    for p in paths:
        rgb, a = load_image(p)
        rgb, a = resize_max(rgb, a, args.resize_max)

        ms = mean_shift_filter(rgb, args.r, args.d, max_iter=args.max_iter, distance=args.distance)
        med = median_shift_filter(
            rgb, args.r, args.d,
            max_iter=args.max_iter, distance=args.distance,
            median_mode=args.median_mode,
            gm_eps=args.gm_eps, gm_max_iter=args.gm_max_iter
        )

        stem = p.stem
        orig_path = outdir / "original" / f"{stem}.png"
        ms_path = outdir / "meanshift" / f"{stem}.png"
        med_path = outdir / "medianshift" / f"{stem}.png"

        save_rgb(orig_path, rgb, alpha=a)
        save_rgb(ms_path, ms, alpha=a)
        save_rgb(med_path, med, alpha=a)

        rows.append((p.name,
                     str(orig_path.relative_to(outdir)),
                     str(ms_path.relative_to(outdir)),
                     str(med_path.relative_to(outdir))))

    params = dict(r=args.r, d=args.d, max_iter=args.max_iter, distance=args.distance, median_mode=args.median_mode)
    write_html(outdir, rows, params)
    print(f"Done. Open: {outdir/'index.html'}")

if __name__ == "__main__":
    main()
```

---

## 5) 参数怎么选（经验向，避免“跑不动/效果不明显”）

* `r`：2~6 常用；窗口越大越慢、越平滑。
* `d`：输入归一化到 [0,1] 后：

  * 对自然图 RGB：0.05~0.15 常用
  * `d` 越小：更“保边缘”、更少混色；太小会几乎不变
* `max_iter`：Wolfram 默认 1；想更平滑可 2~5，但会更慢。([Wolfram 参考文献][1])
* median 版：

  * `component`：速度快、鲁棒性强，但不是严格多元中位数
  * `geometric`：最符合“几何中位数”定义，但慢（建议 `--resize_max`）([维基百科][4])

---

## 参考依据（你提到的“Mathematica MeanShiftFilter”部分我按官方语义对齐）

* Wolfram `MeanShiftFilter` 定义、参数、边界、选项：([Wolfram 参考文献][1])
* Wolfram `MeanShift`（MeanShiftFilter 在邻域上调用它）及其迭代/距离/权重选项背景：([Wolfram 参考文献][2])
* 经典 mean shift filtering 的“联合空间、收敛到 mode、空间/值域带宽”等背景：([Comaniciu][3])
* 几何中位数定义与 Weiszfeld 迭代关系：([维基百科][4])

[1]: https://reference.wolfram.com/language/ref/MeanShiftFilter.html "MeanShiftFilter—Wolfram Documentation"
[2]: https://reference.wolfram.com/language/ref/MeanShift.html "MeanShift—Wolfram Documentation"
[3]: https://comaniciu.net/Papers/MsAnalysis.pdf "iccv99.dvi"
[4]: https://en.wikipedia.org/wiki/Geometric_median?utm_source=chatgpt.com "Geometric median"

