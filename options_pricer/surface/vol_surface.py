"""
surface/vol_surface.py
Volatility surface construction and interpolation.

Takes a sparse grid of (K, T) -> IV observations and returns a smooth surface
via bilinear interpolation (fast) or RBF interpolation (smoother).
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline, RBFInterpolator


# ── data container ────────────────────────────────────────────────────────────

class VolSurface:
    """
    Interpolated implied-volatility surface.

    Parameters
    ----------
    strikes    : array of strike values (x-axis), sorted ascending
    maturities : array of maturity values in years (y-axis), sorted ascending
    iv_grid    : 2-D array of shape (len(strikes), len(maturities))
                 with IV values. NaN entries are filled by nearest neighbour
                 before fitting.
    method     : 'spline' (RectBivariateSpline) or 'rbf' (RBF interpolation)
    """

    def __init__(self, strikes: np.ndarray, maturities: np.ndarray,
                 iv_grid: np.ndarray, method: str = "spline"):
        self.strikes    = np.asarray(strikes,    dtype=float)
        self.maturities = np.asarray(maturities, dtype=float)
        self.iv_grid    = np.asarray(iv_grid,    dtype=float)
        self.method     = method
        self._fit()

    # ── fitting ───────────────────────────────────────────────────────────────

    def _fill_nans(self) -> np.ndarray:
        """Replace NaN cells with nearest valid value (simple row/col fill)."""
        grid = self.iv_grid.copy()
        nan_mask = np.isnan(grid)
        if not nan_mask.any():
            return grid
        # fill each NaN with column mean, then any remaining with row mean
        col_means = np.nanmean(grid, axis=0)
        for j, cm in enumerate(col_means):
            mask = nan_mask[:, j]
            grid[mask, j] = cm if not np.isnan(cm) else np.nanmean(grid)
        return grid

    def _fit(self):
        grid = self._fill_nans()
        if self.method == "spline":
            kx = min(3, len(self.strikes) - 1)
            ky = min(3, len(self.maturities) - 1)
            self._interp = RectBivariateSpline(
                self.strikes, self.maturities, grid, kx=kx, ky=ky
            )
        elif self.method == "rbf":
            # flatten to (N, 2) points + values
            KK, TT = np.meshgrid(self.strikes, self.maturities, indexing="ij")
            pts    = np.column_stack([KK.ravel(), TT.ravel()])
            vals   = grid.ravel()
            self._interp = RBFInterpolator(pts, vals, kernel="thin_plate_spline")
        else:
            raise ValueError("method must be 'spline' or 'rbf'.")

    # ── public ────────────────────────────────────────────────────────────────

    def iv(self, K: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
        """
        Query the surface at arbitrary (K, T) points.

        Parameters
        ----------
        K : strike or array of strikes
        T : maturity or array of maturities (same shape as K for RBF)

        Returns
        -------
        Interpolated IV (scalar or array).
        """
        K = np.atleast_1d(np.asarray(K, dtype=float))
        T = np.atleast_1d(np.asarray(T, dtype=float))

        if self.method == "spline":
            # RectBivariateSpline.ev handles arbitrary points
            result = self._interp.ev(K, T)
        else:
            pts    = np.column_stack([K.ravel(), T.ravel()])
            result = self._interp(pts)

        return np.clip(result, 1e-6, None)   # IV must be positive

    def term_structure(self, K: float) -> tuple[np.ndarray, np.ndarray]:
        """IV vs maturity at a fixed strike."""
        ivs = self.iv(np.full_like(self.maturities, K), self.maturities)
        return self.maturities, ivs

    def smile(self, T: float) -> tuple[np.ndarray, np.ndarray]:
        """IV vs strike at a fixed maturity (volatility smile)."""
        ivs = self.iv(self.strikes, np.full_like(self.strikes, T))
        return self.strikes, ivs

    def grid(self, n_strikes: int = 50,
             n_maturities: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate the surface on a fine regular grid for plotting.

        Returns
        -------
        K_grid  : shape (n_strikes,)
        T_grid  : shape (n_maturities,)
        IV_grid : shape (n_strikes, n_maturities)
        """
        K_fine = np.linspace(self.strikes.min(),    self.strikes.max(),    n_strikes)
        T_fine = np.linspace(self.maturities.min(), self.maturities.max(), n_maturities)
        KK, TT = np.meshgrid(K_fine, T_fine, indexing="ij")
        IV_grid = self.iv(KK.ravel(), TT.ravel()).reshape(n_strikes, n_maturities)
        return K_fine, T_fine, IV_grid


# ── convenience constructor ───────────────────────────────────────────────────

def from_iv_dict(S: float, iv_dict: dict,
                 method: str = "spline") -> VolSurface:
    """
    Build a VolSurface from a dict keyed (K, T) -> IV.

    Parameters
    ----------
    S       : spot price (used to sort strikes relative to ATM)
    iv_dict : {(K, T): iv_value, ...}
    method  : 'spline' or 'rbf'
    """
    strikes    = sorted(set(k for k, _ in iv_dict))
    maturities = sorted(set(t for _, t in iv_dict))
    grid = np.full((len(strikes), len(maturities)), np.nan)
    k_idx = {k: i for i, k in enumerate(strikes)}
    t_idx = {t: j for j, t in enumerate(maturities)}
    for (K, T), iv in iv_dict.items():
        grid[k_idx[K], t_idx[T]] = iv
    return VolSurface(np.array(strikes), np.array(maturities), grid, method)
