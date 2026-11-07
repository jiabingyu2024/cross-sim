#
# Copyright 2017-2026 National Technology & Engineering Solutions of Sandia, LLC
# (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
# Government retains certain rights in this software.
#
# See LICENSE for full license details
#

from .iarray import IArray
from simulator.backend import ComputeBackend
import numpy.typing as npt

xp = ComputeBackend()
_BL_TRIDIAGONAL_KERNEL = None


def _solve_bl_tridiagonal(diagonal, rhs, edges):
    """In-place ladder solve, batched over BLs, windows, and images."""
    n = rhs.shape[1]
    if xp.backend.__name__ == "cupy":
        # One thread per independent ladder avoids thousands of small GPU
        # launches per MVM. Arrays use C order: [BL, row, window, image].
        global _BL_TRIDIAGONAL_KERNEL
        if _BL_TRIDIAGONAL_KERNEL is None:
            _BL_TRIDIAGONAL_KERNEL = xp.RawKernel(r"""
                extern "C" __global__ void solve_bl(
                    double* diagonal, double* rhs, const double* edges,
                    int rows, int samples, int lines) {
                    int line = blockDim.x * blockIdx.x + threadIdx.x;
                    if (line >= lines) return;
                    int bl = line / samples;
                    long long base = (long long)bl * rows * samples + line % samples;
                    for (int i = 1; i < rows; ++i) {
                        long long pos = base + (long long)i * samples;
                        double edge = edges[bl * rows + i - 1];
                        double factor = -edge / diagonal[pos - samples];
                        diagonal[pos] += factor * edge;
                        rhs[pos] -= factor * rhs[pos - samples];
                    }
                    long long last = base + (long long)(rows - 1) * samples;
                    rhs[last] /= diagonal[last];
                    for (int i = rows - 2; i >= 0; --i) {
                        long long pos = base + (long long)i * samples;
                        rhs[pos] = (rhs[pos] + edges[bl * rows + i]
                                    * rhs[pos + samples]) / diagonal[pos];
                    }
                }
            """, "solve_bl")
        samples = rhs.size // (rhs.shape[0] * n)
        lines = rhs.shape[0] * samples
        _BL_TRIDIAGONAL_KERNEL(
            ((lines + 127) // 128,), (128,),
            (diagonal, rhs, edges, n, samples, lines),
        )
    else:
        trailing = (None,) * (rhs.ndim - 2)
        for i in range(1, n):
            edge = edges[:, i - 1][(slice(None), *trailing)]
            factor = -edge / diagonal[:, i - 1]
            diagonal[:, i] += factor * edge
            rhs[:, i] -= factor * rhs[:, i - 1]
        rhs[:, -1] /= diagonal[:, -1]
        for i in range(n - 2, -1, -1):
            edge = edges[:, i][(slice(None), *trailing)]
            rhs[:, i] = (rhs[:, i] + edge * rhs[:, i + 1]) / diagonal[:, i]


class NoninterleavedSeparateSourceArray(IArray):
    """This class models an array where:
    1) The current through the memory cells is sourced from a separate
        interconnect than the input signal, and this interconnect is parallel to
        the output summation column. The input signal is used to
        connect/disconnect the summation column to the current source column.
    2) There is no local interleaving of memory cells for positive and negative
        weights. If differential cells are used, the subtraction occurs outside
        of the array model (e.g. in the peripheral circuits or in digital).
    """

    def __init__(
        self,
        params,
        device=None,
    ) -> None:
        """Initialized a non-interleaved array with seperate sources."""
        super().__init__(params, device=device)
        self.interleaved = False
        self.selected_rows = params.xbar.array.parasitics.selected_rows
        self.rows_max = params.core.rows_max
        self.cols_max = params.core.cols_max

    def solve_mvm(  # noqa:C901
        self,
        vector: npt.NDArray,
        matrix: npt.NDArray,
        row_in: bool = True,
    ) -> npt.NDArray:
        """Calculates the MVM result including parasitic resistance."""
        # Initialize error and number of iterations
        Verr = 1e9
        Niters = 0

        # Compute element-wise voltage drops and currents
        dV0 = self._init_dV(vector, matrix, row_in=row_in)

        # If input is zero, device is gated off and has effectively zero
        # conductance
        active_inputs = xp.abs(dV0) > 1e-9
        if len(dV0.shape) == 4 and len(matrix.shape) != 4:
            matrix = matrix[:, :, None, None] * active_inputs
        else:
            matrix = matrix * active_inputs

        # Create parasitic resistance matrices if they don't exist yet
        if self.Rp_in_mat is None or self.Rp_out_mat is None:
            #   Rp_out is the parasitic resistance of the interconnect that
            #       carries the output partial sums
            #   Rp_in is the parasitic resistance of the interconnect on other
            #       side, supplying or sinking current
            if row_in:
                Rp_in = self.Rp_row_norm
                Rp_out = self.Rp_col_norm
                Rp_in_terminal = self.Rp_row_terminal_norm
                Rp_out_terminal = self.Rp_col_terminal_norm
                Nmax_in = self.rows_max
            else:
                Rp_in = self.Rp_col_norm
                Rp_out = self.Rp_row_norm
                Rp_in_terminal = self.Rp_col_terminal_norm
                Rp_out_terminal = self.Rp_row_terminal_norm
                Nmax_in = self.cols_max

            self.Rp_in_mat = Rp_in * xp.ones((matrix.shape[0], matrix.shape[1]))
            self.Rp_out_mat = Rp_out * xp.ones((matrix.shape[0], matrix.shape[1]))

            # Account for a weight matrix smaller than the array size
            if matrix.shape[1] < Nmax_in:
                N_unselected = Nmax_in - matrix.shape[1]
                if self.selected_rows == "top":
                    # Matrix uses top rows
                    self.Rp_out_mat[:, -1] = Rp_out * (N_unselected + 1)
                elif self.selected_rows == "bottom":
                    # Matrix uses bottom rows
                    self.Rp_in_mat[:, 0] = Rp_in * (N_unselected + 1)

            # Account for terminal resistance
            if Rp_in_terminal > 0 or Rp_out_terminal > 0:
                self.Rp_in_mat[:, 0] += Rp_in_terminal
                self.Rp_out_mat[:, -1] += Rp_out_terminal

        # A linear MVM with resistance only on the summation line is a set of
        # independent resistive ladders. Solve their Kirchhoff equations rather
        # than using a fixed-point iteration, which can diverge at high Rp even
        # though the passive circuit has a unique finite solution.
        if (
            row_in
            and not self.nonlinear_IV
            and bool(xp.all(self.Rp_in_mat == 0))
            and bool(xp.all(self.Rp_out_mat > 0))
        ):
            return self._solve_output_line(matrix, dV0, vector)

        Ires = self._cell_currents(matrix, dV0)
        dV = dV0.copy()

        # Iteratively calculate parasitics and update device currents
        while Verr > self.Verr_th and Niters < self.Niters_max:
            # Calculate parasitic voltage drops
            Isum_col = xp.cumsum(Ires, 1)
            Isum_supply = xp.cumsum(Ires[:, ::-1], 1)[:, ::-1]

            if len(Isum_col.shape) == 4:
                Vdrops_col = xp.cumsum(
                    (self.Rp_out_mat[:, :, None, None] * Isum_col)[:, ::-1], 1
                )[:, ::-1]
                Vdrops_supply = xp.cumsum(
                    self.Rp_in_mat[:, :, None, None] * Isum_supply, 1
                )
            else:
                Vdrops_col = xp.cumsum((self.Rp_out_mat * Isum_col)[:, ::-1], 1)[
                    :, ::-1
                ]
                Vdrops_supply = xp.cumsum(self.Rp_in_mat * Isum_supply, 1)
            Vpar = Vdrops_col + Vdrops_supply
            VerrMat = dV0 - Vpar - dV

            Verr = self._error_metric(VerrMat)
            if Verr < self.Verr_th:
                break

            # Update cell currents for the next iteration
            dV += self.gamma * VerrMat
            Ires = self._cell_currents(matrix, dV)
            Niters += 1

        # The current sum has already been calculated
        Icols = Isum_col[:, -1]
        Icols = self._post_process(Icols, input_dim=len(vector.shape), row_in=row_in)
        if Verr > self.Verr_th:
            raise RuntimeError("Parasitic resistance too high: could not converge!")
        return Icols

    def _solve_output_line(self, matrix, applied_voltage, vector):
        """Solve independent BL ladders with a batched tridiagonal algorithm.

        Matrix axis 1 runs from the far end of a BL to the readout end.
        Rp_out_mat[:, i] connects node i to i+1, with the final segment
        connected to the readout reference. Existing unused-row and terminal
        resistances are retained. Trailing axes are windows and image batches.
        """
        # Double precision avoids cancellation when BL voltages approach the
        # applied voltage. Only the returned currents are cast back to the
        # input dtype, after checking the original circuit's voltage residual.
        conductance = matrix.astype(xp.float64, copy=False)
        voltage = applied_voltage.astype(xp.float64, copy=False)
        resistance = self.Rp_out_mat.astype(xp.float64, copy=False)
        if not bool(xp.isfinite(resistance).all()):
            raise RuntimeError("Non-finite BL segment resistance")
        edges = 1 / resistance
        left_edges = xp.zeros_like(edges)
        left_edges[:, 1:] = edges[:, :-1]
        trailing = (None,) * (voltage.ndim - 2)
        expand = (slice(None), slice(None), *trailing)
        diagonal = conductance + (edges + left_edges)[expand]
        rhs = conductance * voltage
        diagonal = xp.ascontiguousarray(xp.broadcast_to(diagonal, rhs.shape)).copy()
        rhs = xp.ascontiguousarray(rhs).copy()
        _solve_bl_tridiagonal(diagonal, rhs, edges)

        cell_voltage = voltage - rhs
        currents = conductance * cell_voltage
        summed = xp.cumsum(currents, axis=1)
        drops = xp.cumsum((resistance[expand] * summed)[:, ::-1], axis=1)[:, ::-1]
        residual = xp.max(xp.abs(voltage - drops - cell_voltage))
        if not bool(xp.isfinite(residual)) or residual > self.Verr_th:
            raise RuntimeError("Direct BL solver failed circuit residual validation")
        result = summed[:, -1].astype(xp.result_type(matrix.dtype, vector.dtype))
        return self._post_process(result, input_dim=vector.ndim, row_in=True)
