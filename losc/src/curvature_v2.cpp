/**
 * @file
 * @brief Losc curvature version 2 implementation.
 */
#include "curvature.h"
#include "eigen_helper.h"
#include "exception.h"
#include <cmath>

namespace losc {

MatrixXd CurvatureV2::kappa()
{
    // construct absolute overlap under LO.
    MatrixXd S_lo(nlo_, nlo_);
    S_lo.setZero();
    // The LO grid value matrix has dimension of (npts, nlo), which could be
    // very large. To limit the memory usage on LO grid value, we build it by
    // blocks. For now we limit the memory usage to be 1GB.
    const size_t block_size =
        1000ULL * 1000ULL * 1000ULL / sizeof(double) / nlo_;
    size_t nBLK = npts_ / block_size;
    const size_t res = npts_ % block_size;
    if (res != 0) {
        nBLK += 1;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    // loop over all the blocks of grid.
    for (size_t n = 0; n < nBLK; ++n) {
        size_t size = block_size;
        if (n == nBLK - 1 && res != 0)
            size = res;
        // Calculate LO grid value for each block.
        // Formula: grid_lo = grid_ao * C_lo
        // grid_lo: [block_size, nlo]
        // grid_ao: [block_size, nbasis]
        // C_lo: [nbasis, nlo]
        MatrixXd grid_lo_block(size, nlo_);
        grid_lo_block.noalias() =
            grid_basis_value_.block(n * block_size, 0, size, nbasis_) * C_lo_;

        // sum over current block contribution to the LO overlap.
        auto wt = grid_weight_.data() + n * block_size;
        for (size_t ip = 0; ip < size; ++ip) {
            const double wt_value = wt[ip];
            for (size_t i = 0; i < nlo_; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    const double pi = grid_lo_block(ip, i);
                    const double pj = grid_lo_block(ip, j);
                    S_lo(i, j) += wt_value * std::abs(pi * pj);
                }
            }
        }
    }
    mtx_to_symmetric(S_lo, "L");

    // build the curvature version 1.
    CurvatureV1 kappa1_man(dfa_info_, C_lo_, df_pii_, df_Vpq_inverse_,
                           grid_basis_value_, grid_weight_);
    MatrixXd kappa1 = kappa1_man.kappa();

    // build LOSC2 kappa matrix:
    // K2[ij] = erf(tau * S[ij]) * sqrt(abs(K1[ii] * K1[jj])) + erfc(tau *
    // S[ij]) * K][ij]
    using std::abs;
    using std::erf;
    using std::erfc;
    using std::sqrt;
    MatrixXd kappa2(nlo_, nlo_);
    for (size_t i = 0; i < nlo_; ++i) {
        const double K1_ii = kappa1(i, i);
        kappa2(i, i) = K1_ii;
        for (size_t j = 0; j < i; ++j) {
            const double S_ij = S_lo(i, j);
            const double K1_ij = kappa1(i, j);
            const double K1_jj = kappa1(j, j);
            const double f = zeta_ * S_ij;
            kappa2(i, j) = erf(f) * sqrt(abs(K1_ii * K1_jj)) + erfc(f) * K1_ij;
        }
    }
    mtx_to_symmetric(kappa2, "L");

    return kappa2;
}

} // namespace losc
