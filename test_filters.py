import unittest

import numpy as np

import filters


def mean_filter_reference(img: np.ndarray, r: int) -> np.ndarray:
    h, w, c = img.shape
    out = np.empty_like(img)
    for i in range(h):
        i0 = max(0, i - r)
        i1 = min(h, i + r + 1)
        for j in range(w):
            j0 = max(0, j - r)
            j1 = min(w, j + r + 1)
            patch = img[i0:i1, j0:j1, :]
            out[i, j, :] = patch.mean(axis=(0, 1))
    return out


class FiltersTest(unittest.TestCase):
    def test_constant_image_unchanged(self) -> None:
        img = np.ones((16, 16, 3), dtype=np.float32) * 0.37
        out = None
        for out in filters.mean_shift_filter_iter(img, r=2, d=0.1, max_iter=3, backend="numpy", device="cpu"):
            pass
        self.assertIsNotNone(out)
        self.assertTrue(np.allclose(out, img, atol=1e-6, rtol=1e-6))

        out = None
        for out in filters.median_shift_filter_iter(
            img, r=2, d=0.1, max_iter=2, median_mode="component", backend="numpy", device="cpu"
        ):
            pass
        self.assertIsNotNone(out)
        self.assertTrue(np.allclose(out, img, atol=1e-6, rtol=1e-6))

    def test_d_zero_identity(self) -> None:
        rng = np.random.RandomState(0)
        img = rng.rand(12, 10, 3).astype(np.float32)

        ms = next(filters.mean_shift_filter_iter(img, r=3, d=0.0, max_iter=1, backend="numpy", device="cpu"))
        self.assertTrue(np.allclose(ms, img, atol=0, rtol=0))

        med = next(
            filters.median_shift_filter_iter(
                img, r=3, d=0.0, max_iter=1, median_mode="component", backend="numpy", device="cpu"
            )
        )
        self.assertTrue(np.allclose(med, img, atol=0, rtol=0))

    def test_large_d_matches_mean_filter(self) -> None:
        rng = np.random.RandomState(1)
        img = rng.rand(9, 11, 3).astype(np.float32)
        r = 2

        ms = next(filters.mean_shift_filter_iter(img, r=r, d=10.0, max_iter=1, backend="numpy", device="cpu"))
        ref = mean_filter_reference(img, r=r)
        self.assertTrue(np.allclose(ms, ref, atol=1e-6, rtol=1e-6))

    def test_torch_backend_smoke(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not installed")

        img = np.zeros((8, 8, 3), dtype=np.float32)
        img[:, :4, 0] = 1.0

        out = next(filters.mean_shift_filter_iter(img, r=1, d=0.5, max_iter=1, backend="torch", device="cpu"))
        self.assertEqual(out.shape, img.shape)

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            out2 = next(filters.mean_shift_filter_iter(img, r=1, d=0.5, max_iter=1, backend="torch", device="mps"))
            self.assertEqual(out2.shape, img.shape)


if __name__ == "__main__":
    unittest.main()
