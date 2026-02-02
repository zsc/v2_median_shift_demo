from __future__ import annotations

from typing import Iterator, Literal

import numpy as np

DistanceKind = Literal["euclidean", "sqeuclidean", "manhattan", "cosine"]
BackendKind = Literal["auto", "numpy", "torch"]
DeviceKind = Literal["auto", "cpu", "mps"]


def _compute_dist_np(window: np.ndarray, center: np.ndarray, kind: DistanceKind) -> np.ndarray:
    diff = window - center
    if kind == "euclidean":
        return np.sqrt(np.nansum(diff * diff, axis=-1))
    if kind == "sqeuclidean":
        return np.nansum(diff * diff, axis=-1)
    if kind == "manhattan":
        return np.nansum(np.abs(diff), axis=-1)
    if kind == "cosine":
        a = window
        b = center
        num = np.nansum(a * b, axis=-1)
        da = np.sqrt(np.nansum(a * a, axis=-1))
        db = np.sqrt(np.nansum(b * b, axis=-1))
        return 1.0 - (num / (da * db + 1e-12))
    raise ValueError(f"Unknown distance: {kind}")


def _mean_shift_step_np(cur: np.ndarray, r: int, d: float, distance: DistanceKind) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    h, w, _c = cur.shape
    win = 2 * r + 1

    padded = np.pad(cur, pad_width=((r, r), (r, r), (0, 0)), mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, (win, win), axis=(0, 1))  # H W C win win
    windows = windows.transpose(0, 1, 3, 4, 2)  # H W win win C
    center = cur[:, :, None, None, :]  # H W 1 1 C

    valid = ~np.isnan(windows[..., 0])
    dist = _compute_dist_np(windows, center, distance)
    mask = valid & (dist <= d)
    mask[:, :, r, r] = True

    denom = np.sum(mask, axis=(2, 3)).astype(np.float32)
    numer = np.nansum(windows * mask[..., None], axis=(2, 3))
    return (numer / denom[..., None]).astype(np.float32, copy=False)


def mean_shift_filter_iter(
    img: np.ndarray,
    r: int,
    d: float,
    *,
    max_iter: int = 1,
    distance: DistanceKind = "euclidean",
    backend: BackendKind = "auto",
    device: DeviceKind = "auto",
    chunk_pixels: int = 16384,
) -> Iterator[np.ndarray]:
    """
    Yields the image after each outer iteration (t=1..max_iter).

    img: H W C float32 in [0,1]
    """
    cur = np.asarray(img, dtype=np.float32)
    if cur.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {cur.shape}")
    if max_iter < 1:
        return

    if backend == "auto":
        backend = "torch" if _torch_is_available() else "numpy"
    if backend == "torch":
        yield from _mean_shift_filter_iter_torch(
            cur, r, d, max_iter=max_iter, distance=distance, device=device, chunk_pixels=chunk_pixels
        )
        return
    if backend != "numpy":
        raise ValueError(f"Unknown backend: {backend}")

    cur2 = cur.copy()
    for _t in range(1, max_iter + 1):
        cur2 = _mean_shift_step_np(cur2, r, d, distance)
        yield cur2


def _median_shift_step_component_np(cur: np.ndarray, r: int, d: float, distance: DistanceKind) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    h, w, _c = cur.shape
    win = 2 * r + 1

    padded = np.pad(cur, pad_width=((r, r), (r, r), (0, 0)), mode="constant", constant_values=np.nan)
    windows = sliding_window_view(padded, (win, win), axis=(0, 1))  # H W C win win
    windows = windows.transpose(0, 1, 3, 4, 2)  # H W win win C
    center = cur[:, :, None, None, :]

    valid = ~np.isnan(windows[..., 0])
    dist = _compute_dist_np(windows, center, distance)
    mask = valid & (dist <= d)
    mask[:, :, r, r] = True

    masked = np.where(mask[..., None], windows, np.nan)
    return np.nanmedian(masked, axis=(2, 3)).astype(np.float32, copy=False)


def geometric_median_weiszfeld_np(X: np.ndarray, *, eps: float = 1e-5, max_iter: int = 50) -> np.ndarray:
    if X.shape[0] == 1:
        return X[0]
    y = np.nanmean(X, axis=0)
    for _ in range(max_iter):
        diff = X - y[None, :]
        dist = np.linalg.norm(diff, axis=1)
        min_i = int(np.argmin(dist))
        if dist[min_i] < eps:
            return X[min_i]
        w = 1.0 / (dist + 1e-12)
        y_next = (X * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_next - y) < eps:
            return y_next
        y = y_next
    return y


def _median_shift_step_geometric_np(
    cur: np.ndarray,
    r: int,
    d: float,
    distance: DistanceKind,
    *,
    gm_eps: float,
    gm_max_iter: int,
) -> np.ndarray:
    h, w, c = cur.shape
    padded = np.pad(cur, ((r, r), (r, r), (0, 0)), mode="constant", constant_values=np.nan)
    out = np.empty_like(cur)
    win = 2 * r + 1

    for i in range(h):
        for j in range(w):
            patch = padded[i : i + win, j : j + win, :]
            center = cur[i, j, :][None, None, :]
            valid = ~np.isnan(patch[..., 0])
            dist = _compute_dist_np(patch, center, distance)
            mask = valid & (dist <= d)
            mask[r, r] = True

            pts = patch[mask]
            out[i, j, :] = geometric_median_weiszfeld_np(pts, eps=gm_eps, max_iter=gm_max_iter) if pts.size else cur[i, j, :]

    return out


def median_shift_filter_iter(
    img: np.ndarray,
    r: int,
    d: float,
    *,
    max_iter: int = 1,
    distance: DistanceKind = "euclidean",
    median_mode: Literal["geometric", "component"] = "geometric",
    gm_eps: float = 1e-5,
    gm_max_iter: int = 50,
    backend: BackendKind = "auto",
    device: DeviceKind = "auto",
    chunk_pixels: int = 16384,
) -> Iterator[np.ndarray]:
    """
    Yields the image after each outer iteration (t=1..max_iter).

    For `median_mode="geometric"`, this uses a CPU/Numpy reference implementation (slow).
    """
    cur = np.asarray(img, dtype=np.float32)
    if cur.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {cur.shape}")
    if max_iter < 1:
        return

    if median_mode == "geometric":
        cur2 = cur.copy()
        for _t in range(1, max_iter + 1):
            cur2 = _median_shift_step_geometric_np(
                cur2, r, d, distance, gm_eps=gm_eps, gm_max_iter=gm_max_iter
            ).astype(np.float32, copy=False)
            yield cur2
        return

    if median_mode != "component":
        raise ValueError(f"Unknown median_mode: {median_mode}")

    if backend == "auto":
        backend = "torch" if _torch_is_available() else "numpy"
    if backend == "torch":
        try:
            yield from _median_shift_filter_iter_component_torch(
                cur, r, d, max_iter=max_iter, distance=distance, device=device, chunk_pixels=chunk_pixels
            )
            return
        except Exception:
            # Torch backend might not support nanmedian on some devices; fall back to numpy.
            backend = "numpy"
    if backend != "numpy":
        raise ValueError(f"Unknown backend: {backend}")

    cur2 = cur.copy()
    for _t in range(1, max_iter + 1):
        cur2 = _median_shift_step_component_np(cur2, r, d, distance)
        yield cur2


def _torch_is_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _select_torch_device(device: DeviceKind):
    import torch

    if device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Torch MPS requested but not available. Use --device cpu.")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device: {device}")


def _compute_dist_torch(window, center_flat, kind: DistanceKind):
    # window: (B, K, C) ; center_flat: (B, C)
    import torch

    diff = window - center_flat[:, None, :]
    if kind == "euclidean":
        return torch.sqrt(torch.sum(diff * diff, dim=-1))
    if kind == "sqeuclidean":
        return torch.sum(diff * diff, dim=-1)
    if kind == "manhattan":
        return torch.sum(torch.abs(diff), dim=-1)
    if kind == "cosine":
        a = window
        b = center_flat[:, None, :]
        num = torch.sum(a * b, dim=-1)
        da = torch.sqrt(torch.sum(a * a, dim=-1))
        db = torch.sqrt(torch.sum(b * b, dim=-1))
        return 1.0 - (num / (da * db + 1e-12))
    raise ValueError(f"Unknown distance: {kind}")


def _mean_shift_filter_iter_torch(
    img: np.ndarray,
    r: int,
    d: float,
    *,
    max_iter: int,
    distance: DistanceKind,
    device: DeviceKind,
    chunk_pixels: int,
) -> Iterator[np.ndarray]:
    import torch
    import torch.nn.functional as F

    dev = _select_torch_device(device)
    cur = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).to(device=dev)
    cur = cur.permute(2, 0, 1).unsqueeze(0)  # 1 C H W

    for _t in range(1, max_iter + 1):
        padded = F.pad(cur, (r, r, r, r), mode="constant", value=float("nan"))  # 1 C H+2r W+2r
        numer = torch.zeros_like(cur)
        denom = torch.zeros((1, 1, cur.shape[2], cur.shape[3]), device=cur.device, dtype=cur.dtype)

        for dy in range(-r, r + 1):
            y0 = r + dy
            y1 = y0 + cur.shape[2]
            for dx in range(-r, r + 1):
                x0 = r + dx
                x1 = x0 + cur.shape[3]
                neigh = padded[:, :, y0:y1, x0:x1]

                valid = ~torch.isnan(neigh[:, :1, :, :])
                if distance == "euclidean":
                    diff = neigh - cur
                    dist = torch.sqrt(torch.sum(diff * diff, dim=1, keepdim=True))
                elif distance == "sqeuclidean":
                    diff = neigh - cur
                    dist = torch.sum(diff * diff, dim=1, keepdim=True)
                elif distance == "manhattan":
                    dist = torch.sum(torch.abs(neigh - cur), dim=1, keepdim=True)
                elif distance == "cosine":
                    num = torch.sum(neigh * cur, dim=1, keepdim=True)
                    da = torch.sqrt(torch.sum(neigh * neigh, dim=1, keepdim=True))
                    db = torch.sqrt(torch.sum(cur * cur, dim=1, keepdim=True))
                    dist = 1.0 - (num / (da * db + 1e-12))
                else:
                    raise ValueError(f"Unknown distance: {distance}")

                mask = valid & (dist <= d)
                numer = numer + torch.where(mask, neigh, 0.0)
                denom = denom + mask.to(cur.dtype)

        # Center pixel always included (dy=dx=0), denom >= 1 everywhere.
        cur = numer / denom
        yield cur.squeeze(0).permute(1, 2, 0).detach().to("cpu").numpy()


def _median_shift_filter_iter_component_torch(
    img: np.ndarray,
    r: int,
    d: float,
    *,
    max_iter: int,
    distance: DistanceKind,
    device: DeviceKind,
    chunk_pixels: int,
) -> Iterator[np.ndarray]:
    import torch
    import torch.nn.functional as F

    dev = _select_torch_device(device)
    cur = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).to(device=dev)
    cur = cur.permute(2, 0, 1).unsqueeze(0)  # 1 C H W

    _n, c, h, w = cur.shape
    win = 2 * r + 1
    k = win * win
    nan = torch.tensor(float("nan"), device=cur.device, dtype=cur.dtype)

    for _t in range(1, max_iter + 1):
        padded = F.pad(cur, (r, r, r, r), mode="constant", value=float("nan"))  # 1 C H+2r W+2r
        out = torch.empty_like(cur)

        # Chunk along height to control memory: (K * C * chunk_h * W).
        chunk_h = max(1, int(chunk_pixels // max(1, w)))
        for y in range(0, h, chunk_h):
            y_end = min(h, y + chunk_h)
            center = cur[:, :, y:y_end, :]  # 1 C ch W
            ch = y_end - y

            vals = torch.empty((k, 1, c, ch, w), device=cur.device, dtype=cur.dtype)
            idx = 0
            for dy in range(-r, r + 1):
                y0 = r + y + dy
                y1 = y0 + ch
                for dx in range(-r, r + 1):
                    x0 = r + dx
                    x1 = x0 + w
                    neigh = padded[:, :, y0:y1, x0:x1]  # 1 C ch W

                    valid = ~torch.isnan(neigh[:, :1, :, :])
                    if distance == "euclidean":
                        diff = neigh - center
                        dist = torch.sqrt(torch.sum(diff * diff, dim=1, keepdim=True))
                    elif distance == "sqeuclidean":
                        diff = neigh - center
                        dist = torch.sum(diff * diff, dim=1, keepdim=True)
                    elif distance == "manhattan":
                        dist = torch.sum(torch.abs(neigh - center), dim=1, keepdim=True)
                    elif distance == "cosine":
                        num = torch.sum(neigh * center, dim=1, keepdim=True)
                        da = torch.sqrt(torch.sum(neigh * neigh, dim=1, keepdim=True))
                        db = torch.sqrt(torch.sum(center * center, dim=1, keepdim=True))
                        dist = 1.0 - (num / (da * db + 1e-12))
                    else:
                        raise ValueError(f"Unknown distance: {distance}")

                    mask = valid & (dist <= d)
                    vals[idx] = torch.where(mask, neigh, nan)
                    idx += 1

            med = torch.nanmedian(vals, dim=0)
            out[:, :, y:y_end, :] = med.values if hasattr(med, "values") else med[0]

        cur = out
        yield cur.squeeze(0).permute(1, 2, 0).detach().to("cpu").numpy()
