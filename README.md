# Two-loop MSSM RGE Solver
This Python code solves the two-loop renormalization group equations for the MSSM, including soft-SUSY breaking couplings and masses, as well as $\tan(\beta)$ evolution (in the $R_{\xi=1}$ Feynman gauge, derived from arXiv:hep-ph/0112251), based on user-input GUT-scale boundary conditions.

In the first version of this program, the Yukawa couplings, soft trilinear couplings, and scalar masses are treated as $3\times3$ matrices in generation space. Gaugino masses are also assumed to be real. However, these matrices are assumed to be *_real_* as well as *_diagonal_*. The first two generations are *_not_* ignored, and some boundary conditions are obtained from FlexibleSUSY and Isajet. All computations are done in the $\overline{\text{DR}}$ renormalization scheme.
