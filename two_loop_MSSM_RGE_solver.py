# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 15:52:47 2022

Two-Loop Renormalization Group Equation solver for the MSSM for a user-supplied
    set of input parameters.

@author: Dakotah Martinez
"""

from scipy.integrate import solve_ivp
import numpy as np
import sympy
from sympy import log, sqrt, nsolve
from sympy.abc import x
import os

##### Constants #####

loop_fac = 1 / (16 * np.power(np.pi, 2))
loop_fac_sq = np.power(loop_fac, 2)
b_1l=[33/5, 1, -3]
b_2l=[[199/25, 27/5, 88/5], [9/5, 25, 24], [11/5, 9, 14]]
c_2l=[[26/5, 14/5, 18/5], [6, 6, 2], [4, 4, 0]]

##### RGEs #####
def my_RGE_solver(GUT_BCs):
    def my_odes(t, x):
        """
        Define one-loop RGEs for soft terms.

        Parameters
        ----------
        x : Numerical solutions to RGEs. The order of entries in x is:
                (0: g1, 1: g2, 2: g3, 3: M1, 4: M2, 5: M3, 6: mu, 7: yt, 8: yc,
                 9: yu, 10: yb, 11: ys, 12: yd, 13: ytau, 14: ymu, 15: ye,
                 16: at, 17: ac, 18: au, 19: ab, 20: as, 21: ad, 22: atau,
                 23: amu, 24: ae, 25: b, 26: mHu^2, 27: mHd^2, 28: mQ1^2,
                 29: mQ2^2, 30: mQ3^2, 31: mL1^2, 32: mL2^2, 33: mL3^2,
                 34: mU1^2, 35: mU2^2, 36: mU3^2, 37: mD1^2, 38: mD2^2,
                 39: mD3^2, 40: mE1^2, 41: mE2^2, 42: mE3^2, 43: tanb)
        t : t = Q values for numerical solutions.

        Returns
        -------
        Return all soft RGEs.

        """
        # Unification scale is acquired from running a BM point through
        # Isajet, then GUT scale boundary conditions for gauge couplings and
        # Yukawas are acquired from
        # FlexibleSUSY so that all three generations of Yukawas (assumed
        # to be diagonalized) are accounted for. A universal boundary condition
        # is used for soft scalar trilinear couplings a_i=y_i*A_i.
        # The soft b^(ij) mass^2 term is defined as b=B*mu.
        # Scalar mass matrices will also be written in diagonalized form such
        # that, e.g., mQ^2=((mQ1^2,0,0),(0,mQ2^2,0),(0,0,mQ3^2)).

        # Define all parameters in terms of solution vector x
        g1_val = x[0]
        g2_val = x[1]
        g3_val = x[2]
        M1_val = x[3]
        M2_val = x[4]
        M3_val = x[5]
        mu_val = x[6]
        yt_val = x[7]
        yc_val = x[8]
        yu_val = x[9]
        yb_val = x[10]
        ys_val = x[11]
        yd_val = x[12]
        ytau_val = x[13]
        ymu_val = x[14]
        ye_val = x[15]
        at_val = x[16]
        ac_val = x[17]
        au_val = x[18]
        ab_val = x[19]
        as_val = x[20]
        ad_val = x[21]
        atau_val = x[22]
        amu_val = x[23]
        ae_val = x[24]
        b_val = x[25]
        mHu_sq_val = x[26]
        mHd_sq_val = x[27]
        mQ1_sq_val = x[28]
        mQ2_sq_val = x[29]
        mQ3_sq_val = x[30]
        mL1_sq_val = x[31]
        mL2_sq_val = x[32]
        mL3_sq_val = x[33]
        mU1_sq_val = x[34]
        mU2_sq_val = x[35]
        mU3_sq_val = x[36]
        mD1_sq_val = x[37]
        mD2_sq_val = x[38]
        mD3_sq_val = x[39]
        mE1_sq_val = x[40]
        mE2_sq_val = x[41]
        mE3_sq_val = x[42]
        tanb_val = x[43]

        ##### Gauge couplings and gaugino masses #####
        # 1 loop parts
        dg1_dt_1l = (1 / t) * loop_fac * b[0] * np.power(g1_val, 3)

        dg2_dt_1l = (1 / t) * loop_fac * b[1] * np.power(g2_val, 3)

        dg3_dt_1l = (1 / t) * loop_fac * b[2] * np.power(g3_val, 3)

        dM1_dt_1l = (2 / t) * loop_fac * b[0] * np.power(g1_val, 2) * M1_val

        dM2_dt_1l = (2 / t) * loop_fac * b[1] * np.power(g2_val, 2) * M2_val

        dM3_dt_1l = (2 / t) * loop_fac * b[2] * np.power(g3_val, 2) * M3_val

        # 2 loop parts
        dg1_dt_2l = (1 / t) * loop_fac_sq * np.power(g1_val, 3)\
            * ((b_2l[0][0] * np.power(g1_val, 2))
               + (b_2l[0][1] * np.power(g2_val, 2))
               + (b_2l[0][2] * np.power(g3_val, 2))
               - (c_2l[0][0] * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))
               - (c_2l[0][1] * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))
               - (c_2l[0][2] * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2))))

        dg2_dt_2l = (1 / t) * loop_fac_sq * np.power(g2_val, 3)\
            * ((b_2l[1][0] * np.power(g1_val, 2))
               + (b_2l[1][1] * np.power(g2_val, 2))
               + (b_2l[1][2] * np.power(g3_val, 2))
               - (c_2l[1][0] * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))
               - (c_2l[1][1] * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))
               - (c_2l[1][2] * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2))))

        dg3_dt_2l = (1 / t) * loop_fac_sq * np.power(g3_val, 3)\
            * ((b_2l[2][0] * np.power(g1_val, 2))
               + (b_2l[2][1] * np.power(g2_val, 2))
               + (b_2l[2][2] * np.power(g3_val, 2))
               - (c_2l[2][0] * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))
               - (c_2l[2][1] * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))
               - (c_2l[2][2] * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2))))

        dM1_dt_2l = 2 * np.power(g1_val, 2)\
            * (((b_2l[0][0] * np.power(g1_val, 2) * (M1_val + M1_val))
                + (b_2l[0][1] * np.power(g2_val, 2) * (M1_val + M2_val))
                + (b_2l[0][2] * np.power(g3_val, 2) * (M1_val + M3_val)))
               + ((c_2l[0][0] * (((yt_val * at_val) + (yc_val * ac_val)
                                  + (yu_val * au_val))
                                 - (M1_val * (np.power(yt_val, 2)
                                              + np.power(yc_val, 2)
                                              + np.power(yu_val, 2))))))
               + ((c_2l[0][1] * (((yb_val * ab_val) + (ys_val * as_val)
                                  + (yd_val * ad_val))
                                 - (M1_val * (np.power(yb_val, 2)
                                              + np.power(ys_val, 2)
                                              + np.power(yd_val, 2))))))
               + ((c_2l[0][2] * (((ytau_val * atau_val) + (ymu_val * amu_val)
                                  + (ye_val * ae_val))
                                 - (M1_val * (np.power(ytau_val, 2)
                                              + np.power(ymu_val, 2)
                                              + np.power(ye_val, 2)))))))

        dM2_dt_2l = 2 * np.power(g2_val, 2)\
            * (((b_2l[1][0] * np.power(g1_val, 2) * (M2_val + M1_val))
                + (b_2l[1][1] * np.power(g2_val, 2) * (M2_val + M2_val))
                + (b_2l[1][2] * np.power(g3_val, 2) * (M2_val + M3_val)))
               + ((c_2l[1][0] * (((yt_val * at_val) + (yc_val * ac_val)
                                  + (yu_val * au_val))
                                 - (M2_val * (np.power(yt_val, 2)
                                              + np.power(yc_val, 2)
                                              + np.power(yu_val, 2))))))
               + ((c_2l[1][1] * (((yb_val * ab_val) + (ys_val * as_val)
                                  + (yd_val * ad_val))
                                 - (M2_val * (np.power(yb_val, 2)
                                              + np.power(ys_val, 2)
                                              + np.power(yd_val, 2))))))
               + ((c_2l[1][2] * (((ytau_val * atau_val) + (ymu_val * amu_val)
                                  + (ye_val * ae_val))
                                 - (M2_val * (np.power(ytau_val, 2)
                                              + np.power(ymu_val, 2)
                                              + np.power(ye_val, 2)))))))

        dM3_dt_2l = 2 * np.power(g3_val, 2)\
            * (((b_2l[2][0] * np.power(g1_val, 2) * (M3_val + M1_val))
                + (b_2l[2][1] * np.power(g2_val, 2) * (M3_val + M2_val))
                + (b_2l[2][2] * np.power(g3_val, 2) * (M3_val + M3_val)))
               + ((c_2l[2][0] * (((yt_val * at_val) + (yc_val * ac_val)
                                  + (yu_val * au_val))
                                 - (M3_val * (np.power(yt_val, 2)
                                              + np.power(yc_val, 2)
                                              + np.power(yu_val, 2))))))
               + ((c_2l[2][1] * (((yb_val * ab_val) + (ys_val * as_val)
                                  + (yd_val * ad_val))
                                 - (M3_val * (np.power(yb_val, 2)
                                              + np.power(ys_val, 2)
                                              + np.power(yd_val, 2))))))
               + ((c_2l[2][2] * (((ytau_val * atau_val) + (ymu_val * amu_val)
                                  + (ye_val * ae_val))
                                 - (M3_val * (np.power(ytau_val, 2)
                                              + np.power(ymu_val, 2)
                                              + np.power(ye_val, 2)))))))

        # Total gauge and gaugino mass beta functions
        dg1_dt = (1 / t) * ((loop_fac * dg1_dt_1l)
                            + (loop_fac_sq * dg1_dt_2l))

        dg2_dt = (1 / t) * ((loop_fac * dg2_dt_1l)
                            + (loop_fac_sq * dg2_dt_2l))

        dg3_dt = (1 / t) * ((loop_fac * dg3_dt_1l)
                            + (loop_fac_sq * dg3_dt_2l))

        dM1_dt = (2 / t) * ((loop_fac * dM1_dt_1l)
                             + (loop_fac_sq * dM1_dt_2l))

        dM2_dt = (2 / t) * ((loop_fac * dM2_dt_1l)
                             + (loop_fac_sq * dM2_dt_2l))

        dM3_dt = (2 / t) * ((loop_fac * dM3_dt_1l)
                             + (loop_fac_sq * dM3_dt_2l))

        ##### Higgsino mass parameter mu #####
        # 1 loop part
        dmu_dt_1l = mu_val\
            * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                     + np.power(yu_val, 2) + np.power(yb_val, 2)
                     + np.power(ys_val, 2) + np.power(yd_val, 2)))
               + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                  + np.power(ye_val, 2))
               - (3 * np.power(g2_val, 2))
               - ((3 / 5) * np.power(g1_val, 2)))

        # 2 loop part
        dmu_dt_2l = mu_val\
            * ((-3 * ((3 * (np.power(yt_val, 4) + np.power(yc_val, 4)
                     + np.power(yu_val, 4) + np.power(yb_val, 4)
                     + np.power(ys_val, 4) + np.power(yd_val, 4)))
               + (np.power(ytau_val, 4) + np.power(ymu_val, 4)
                  + np.power(ye_val, 4))
               + (2 * ((np.power(yt_val, 2) * np.power(yb_val, 2))
                       + (np.power(yc_val, 2) * np.power(ys_val, 2))
                       + (np.power(yu_val, 2) * np.power(yd_val, 2))))))
               + (((16 * np.power(g3_val, 2)) + (4 * np.power(g1_val, 2) / 5))
                  * (np.power(yt_val, 2) + np.power(yc_val, 2)
                     + np.power(yu_val, 2)))
               + (((16 * np.power(g3_val, 2)) - (2 * np.power(g1_val, 2) / 5))
                  * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + ((6 / 5) * np.power(g1_val, 2)
                  * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                     + np.power(ye_val, 2)))
               + ((15 / 2) * np.power(g2_val, 4))
               + ((9 / 5) * np.power(g1_val, 2) * np.power(g2_val, 2))
               + ((207 / 50) * np.power(g1_val, 4)))

        # Total mu beta function
        dmu_dt = (1 / t) * ((loop_fac * dmu_dt_1l)
                            + (loop_fac_sq * dmu_dt_2l))

        ##### Yukawa couplings for all 3 generations, assumed diagonalized#####
        # 1 loop parts
        dyt_dt_1l = yt_val\
            * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                     + np.power(yu_val, 2)))
               + (3 * (np.power(yt_val, 2)))
               + np.power(yb_val, 2) - ((16 / 3) * np.power(g3_val, 2))
               - (3 * np.power(g2_val, 2)) - ((13 / 15) * np.power(g1_val, 2)))

        dyc_dt_1l = yc_val\
            * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                     + np.power(yu_val, 2)))
               + (3 * (np.power(yc_val, 2)))
               + np.power(ys_val, 2) - ((16 / 3) * np.power(g3_val, 2))
               - (3 * np.power(g2_val, 2)) - ((13 / 15) * np.power(g1_val, 2)))

        dyu_dt_1l = yu_val\
            * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                     + np.power(yu_val, 2)))
               + (3 * (np.power(yu_val, 2)))
               + np.power(yd_val, 2) - ((16 / 3) * np.power(g3_val, 2))
               - (3 * np.power(g2_val, 2)) - ((13 / 15) * np.power(g1_val, 2)))

        dyb_dt_1l = yb_val\
            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                  + np.power(ye_val, 2))
               + (3 * (np.power(yb_val, 2))) + np.power(yt_val, 2)
               - ((16 / 3) * np.power(g3_val, 2))
               - (3 * np.power(g2_val, 2)) - ((7 / 15) * np.power(g1_val, 2)))

        dys_dt_1l = ys_val\
            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                  + np.power(ye_val, 2))
               + (3 * (np.power(ys_val, 2))) + np.power(yc_val, 2)
               - ((16 / 3) * np.power(g3_val, 2))
               - (3 * np.power(g2_val, 2)) - ((7 / 15) * np.power(g1_val, 2)))

        dyd_dt_1l = yd_val\
            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                  + np.power(ye_val, 2))
               + (3 * (np.power(yd_val, 2))) + np.power(yu_val, 2)
               - ((16 / 3) * np.power(g3_val, 2))
               - (3 * np.power(g2_val, 2)) - ((7 / 15) * np.power(g1_val, 2)))

        dytau_dt_1l = ytau_val\
            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                  + np.power(ye_val, 2))
               + (3 * (np.power(ytau_val, 2)))
               - (3 * np.power(g2_val, 2)) - ((9 / 5) * np.power(g1_val, 2)))

        dymu_dt_1l = ymu_val\
            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                  + np.power(ye_val, 2))
               + (3 * (np.power(ymu_val, 2)))
               - (3 * np.power(g2_val, 2)) - ((9 / 5) * np.power(g1_val, 2)))

        dye_dt_1l = ye_val\
            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                  + np.power(ye_val, 2))
               + (3 * (np.power(ye_val, 2)))
               - (3 * np.power(g2_val, 2)) - ((9 / 5) * np.power(g1_val, 2)))

        # 2 loop parts
        dyt_dt_2l = yt_val\
            * (((-3) * ((3 * (np.power(yt_val, 4) + np.power(yc_val, 4)
                             + np.power(yu_val, 4)))
                       + (np.power(yt_val, 2) * np.power(yb_val, 2))
                       + (np.power(yc_val, 2) * np.power(ys_val, 2))
                       + (np.power(yu_val, 2) * np.power(yd_val, 2))))
               - (np.power(yb_val, 2) * ((3 * (np.power(yb_val, 2)
                                               + np.power(ys_val, 2)
                                               + np.power(yd_val, 2)))
                                         + (np.power(ytau_val, 2)
                                            + np.power(ymu_val, 2)
                                            + np.power(ye_val, 2))))
               - (9 * np.power(yt_val, 2) * (np.power(yt_val, 2)
                                             + np.power(yc_val, 2)
                                             + np.power(yu_val, 2)))
               - (4 * np.power(yt_val, 4)) - (2 * np.power(yb_val, 4))
               - (2 * np.power(yb_val, 2) * np.power(yt_val, 2))
               + (((16 * np.power(g3_val, 2))
                   + ((4 / 5) * np.power(g1_val, 2)))
                  * (np.power(yt_val, 2) + np.power(yc_val, 2)
                     + np.power(yu_val, 2)))
               + (((6 * np.power(g2_val, 2))
                  + ((2 / 5) * np.power(g1_val, 2))) * np.power(yt_val, 2))
               + ((2 / 5) * np.power(g1_val, 2) * np.power(yb_val, 2))
               - ((16 / 9) * np.power(g3_val, 4))
               + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
               + ((136 / 45) * np.power(g3_val, 2) * np.power(g1_val, 2))
               + ((15 / 2) * np.power(g2_val, 4))
               + (np.power(g2_val, 2) * np.power(g1_val, 2))
               + ((2743 / 450) * np.power(g1_val, 4)))

        dyc_dt_2l = yc_val\
            * (((-3) * ((3 * (np.power(yt_val, 4) + np.power(yc_val, 4)
                             + np.power(yu_val, 4)))
                       + (np.power(yt_val, 2) * np.power(yb_val, 2))
                       + (np.power(yc_val, 2) * np.power(ys_val, 2))
                       + (np.power(yu_val, 2) * np.power(yd_val, 2))))
               - (np.power(ys_val, 2) * ((3 * (np.power(yb_val, 2)
                                               + np.power(ys_val, 2)
                                               + np.power(yd_val, 2)))
                                         + (np.power(ytau_val, 2)
                                            + np.power(ymu_val, 2)
                                            + np.power(ye_val, 2))))
               - (9 * np.power(yc_val, 2) * (np.power(yt_val, 2)
                                             + np.power(yc_val, 2)
                                             + np.power(yu_val, 2)))
               - (4 * np.power(yc_val, 4)) - (2 * np.power(ys_val, 4))
               - (2 * np.power(ys_val, 2) * np.power(yc_val, 2))
               + (((16 * np.power(g3_val, 2))
                   + ((4 / 5) * np.power(g1_val, 2)))
                  * (np.power(yt_val, 2) + np.power(yc_val, 2)
                     + np.power(yu_val, 2)))
               + (((6 * np.power(g2_val, 2))
                  + ((2 / 5) * np.power(g1_val, 2))) * np.power(yc_val, 2))
               + ((2 / 5) * np.power(g1_val, 2) * np.power(ys_val, 2))
               - ((16 / 9) * np.power(g3_val, 4))
               + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
               + ((136 / 45) * np.power(g3_val, 2) * np.power(g1_val, 2))
               + ((15 / 2) * np.power(g2_val, 4))
               + (np.power(g2_val, 2) * np.power(g1_val, 2))
               + ((2743 / 450) * np.power(g1_val, 4)))

        dyu_dt_2l = yu_val\
            * (((-3) * ((3 * (np.power(yt_val, 4) + np.power(yc_val, 4)
                             + np.power(yu_val, 4)))
                       + (np.power(yt_val, 2) * np.power(yb_val, 2))
                       + (np.power(yc_val, 2) * np.power(ys_val, 2))
                       + (np.power(yu_val, 2) * np.power(yd_val, 2))))
               - (np.power(yd_val, 2) * ((3 * (np.power(yb_val, 2)
                                               + np.power(ys_val, 2)
                                               + np.power(yd_val, 2)))
                                         + (np.power(ytau_val, 2)
                                            + np.power(ymu_val, 2)
                                            + np.power(ye_val, 2))))
               - (9 * np.power(yu_val, 2) * (np.power(yt_val, 2)
                                             + np.power(yc_val, 2)
                                             + np.power(yu_val, 2)))
               - (4 * np.power(yu_val, 4)) - (2 * np.power(yd_val, 4))
               - (2 * np.power(yd_val, 2) * np.power(yu_val, 2))
               + (((16 * np.power(g3_val, 2))
                   + ((4 / 5) * np.power(g1_val, 2)))
                  * (np.power(yt_val, 2) + np.power(yc_val, 2)
                     + np.power(yu_val, 2)))
               + (((6 * np.power(g2_val, 2))
                  + ((2 / 5) * np.power(g1_val, 2))) * np.power(yu_val, 2))
               + ((2 / 5) * np.power(g1_val, 2) * np.power(yd_val, 2))
               - ((16 / 9) * np.power(g3_val, 4))
               + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
               + ((136 / 45) * np.power(g3_val, 2) * np.power(g1_val, 2))
               + ((15 / 2) * np.power(g2_val, 4))
               + (np.power(g2_val, 2) * np.power(g1_val, 2))
               + ((2743 / 450) * np.power(g1_val, 4)))

        dyb_dt_2l = yb_val\
            * (((-3) * ((3 * (np.power(yb_val, 4) + np.power(ys_val, 4)
                             + np.power(yd_val, 4)))
                       + (np.power(yt_val, 2) * np.power(yb_val, 2))
                       + (np.power(yc_val, 2) * np.power(ys_val, 2))
                       + (np.power(yu_val, 2) * np.power(yd_val, 2))
                       + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                       + np.power(ye_val, 4)))
               - (3 * np.power(yt_val, 2) * (np.power(yt_val, 2)
                                             + np.power(yc_val, 2)
                                             + np.power(yu_val, 2)))
               - (3 * np.power(yb_val, 2) * ((3 * (np.power(yb_val, 2)
                                                   + np.power(ys_val, 2)
                                                   + np.power(yd_val, 2)))
                                             + np.power(ytau_val, 2)
                                             + np.power(ymu_val, 2)
                                             + np.power(ye_val, 2)))
               - (4 * np.power(yb_val, 4)) - (2 * np.power(yt_val, 4))
               - (2 * np.power(yt_val, 2) * np.power(yb_val, 2))
               + (((16 * np.power(g3_val, 2))
                   - ((2 / 5) * np.power(g1_val, 2)))
                  * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + ((6 / 5) * np.power(g1_val, 2) * (np.power(ytau_val, 2)
                                                   + np.power(ymu_val, 2)
                                                   + np.power(ye_val, 2)))
               + ((4 / 5) * np.power(g1_val, 2) * np.power(yt_val, 2))
               + (np.power(yb_val, 2) * ((6 * np.power(g2_val, 2))
                                         + ((4 / 5) * np.power(g1_val, 2))))
               - ((16 / 9) * np.power(g3_val, 4))
               + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
               + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
               + ((15 / 2) * np.power(g2_val, 4))
               + (np.power(g2_val, 2) * np.power(g1_val, 2))
               + ((287 / 90) * np.power(g1_val, 4)))

        dys_dt_2l = ys_val\
            * (((-3) * ((3 * (np.power(yb_val, 4) + np.power(ys_val, 4)
                             + np.power(yd_val, 4)))
                       + (np.power(yt_val, 2) * np.power(yb_val, 2))
                       + (np.power(yc_val, 2) * np.power(ys_val, 2))
                       + (np.power(yu_val, 2) * np.power(yd_val, 2))
                       + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                       + np.power(ye_val, 4)))
               - (3 * np.power(yc_val, 2) * (np.power(yt_val, 2)
                                             + np.power(yc_val, 2)
                                             + np.power(yu_val, 2)))
               - (3 * np.power(ys_val, 2) * ((3 * (np.power(yb_val, 2)
                                                   + np.power(ys_val, 2)
                                                   + np.power(yd_val, 2)))
                                             + np.power(ytau_val, 2)
                                             + np.power(ymu_val, 2)
                                             + np.power(ye_val, 2)))
               - (4 * np.power(ys_val, 4)) - (2 * np.power(yc_val, 4))
               - (2 * np.power(yc_val, 2) * np.power(ys_val, 2))
               + (((16 * np.power(g3_val, 2))
                   - ((2 / 5) * np.power(g1_val, 2)))
                  * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + ((6 / 5) * np.power(g1_val, 2) * (np.power(ytau_val, 2)
                                                   + np.power(ymu_val, 2)
                                                   + np.power(ye_val, 2)))
               + ((4 / 5) * np.power(g1_val, 2) * np.power(yc_val, 2))
               + (np.power(ys_val, 2) * ((6 * np.power(g2_val, 2))
                                         + ((4 / 5) * np.power(g1_val, 2))))
               - ((16 / 9) * np.power(g3_val, 4))
               + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
               + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
               + ((15 / 2) * np.power(g2_val, 4))
               + (np.power(g2_val, 2) * np.power(g1_val, 2))
               + ((287 / 90) * np.power(g1_val, 4)))

        dyd_dt_2l = yd_val\
            * (((-3) * ((3 * (np.power(yb_val, 4) + np.power(ys_val, 4)
                             + np.power(yd_val, 4)))
                       + (np.power(yt_val, 2) * np.power(yb_val, 2))
                       + (np.power(yc_val, 2) * np.power(ys_val, 2))
                       + (np.power(yu_val, 2) * np.power(yd_val, 2))
                       + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                       + np.power(ye_val, 4)))
               - (3 * np.power(yu_val, 2) * (np.power(yt_val, 2)
                                             + np.power(yc_val, 2)
                                             + np.power(yu_val, 2)))
               - (3 * np.power(yd_val, 2) * ((3 * (np.power(yb_val, 2)
                                                   + np.power(ys_val, 2)
                                                   + np.power(yd_val, 2)))
                                             + np.power(ytau_val, 2)
                                             + np.power(ymu_val, 2)
                                             + np.power(ye_val, 2)))
               - (4 * np.power(yd_val, 4)) - (2 * np.power(yu_val, 4))
               - (2 * np.power(yd_val, 2) * np.power(yu_val, 2))
               + (((16 * np.power(g3_val, 2))
                   - ((2 / 5) * np.power(g1_val, 2)))
                  * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + ((6 / 5) * np.power(g1_val, 2) * (np.power(ytau_val, 2)
                                                   + np.power(ymu_val, 2)
                                                   + np.power(ye_val, 2)))
               + ((4 / 5) * np.power(g1_val, 2) * np.power(yu_val, 2))
               + (np.power(yd_val, 2) * ((6 * np.power(g2_val, 2))
                                         + ((4 / 5) * np.power(g1_val, 2))))
               - ((16 / 9) * np.power(g3_val, 4))
               + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
               + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
               + ((15 / 2) * np.power(g2_val, 4))
               + (np.power(g2_val, 2) * np.power(g1_val, 2))
               + ((287 / 90) * np.power(g1_val, 4)))

        dytau_dt_2l = ytau_val\
            * (((-3) * ((3 * (np.power(yb_val, 4) + np.power(ys_val, 4)
                             + np.power(yd_val, 4)))
                       + (np.power(yt_val, 2) * np.power(yb_val, 2))
                       + (np.power(yc_val, 2) * np.power(ys_val, 2))
                       + (np.power(yu_val, 2) * np.power(yd_val, 2))
                       + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                       + np.power(ye_val, 4)))
               - (3 * np.power(ytau_val, 2) * ((3 * (np.power(yb_val, 2)
                                                     + np.power(ys_val, 2)
                                                     + np.power(yd_val, 2)))
                                               + np.power(ytau_val, 2)
                                               + np.power(ymu_val, 2)
                                               + np.power(ye_val, 2)))
               - (4 * np.power(ytau_val, 4))
               + (((16 * np.power(g3_val, 2))
                   - ((2 / 5) * np.power(g1_val, 2)))
                  * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + ((6 / 5) * np.power(g1_val, 2) * (np.power(ytau_val, 2)
                                                   + np.power(ymu_val, 2)
                                                   + np.power(ye_val, 2)))
               + (6 * np.power(g2_val, 2) * np.power(ytau_val, 2))
               + ((15 / 2) * np.power(g2_val, 4))
               + ((9 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2))
               + ((27 / 2) * np.power(g1_val, 4)))

        dymu_dt_2l = ymu_val\
            * (((-3) * ((3 * (np.power(yb_val, 4) + np.power(ys_val, 4)
                             + np.power(yd_val, 4)))
                       + (np.power(yt_val, 2) * np.power(yb_val, 2))
                       + (np.power(yc_val, 2) * np.power(ys_val, 2))
                       + (np.power(yu_val, 2) * np.power(yd_val, 2))
                       + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                       + np.power(ye_val, 4)))
               - (3 * np.power(ymu_val, 2) * ((3 * (np.power(yb_val, 2)
                                                    + np.power(ys_val, 2)
                                                    + np.power(yd_val, 2)))
                                              + np.power(ytau_val, 2)
                                              + np.power(ymu_val, 2)
                                              + np.power(ye_val, 2)))
               - (4 * np.power(ymu_val, 4))
               + (((16 * np.power(g3_val, 2))
                   - ((2 / 5) * np.power(g1_val, 2)))
                  * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + ((6 / 5) * np.power(g1_val, 2) * (np.power(ytau_val, 2)
                                                   + np.power(ymu_val, 2)
                                                   + np.power(ye_val, 2)))
               + (6 * np.power(g2_val, 2) * np.power(ymu_val, 2))
               + ((15 / 2) * np.power(g2_val, 4))
               + ((9 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2))
               + ((27 / 2) * np.power(g1_val, 4)))

        dye_dt_2l = ye_val\
            * (((-3) * ((3 * (np.power(yb_val, 4) + np.power(ys_val, 4)
                             + np.power(yd_val, 4)))
                       + (np.power(yt_val, 2) * np.power(yb_val, 2))
                       + (np.power(yc_val, 2) * np.power(ys_val, 2))
                       + (np.power(yu_val, 2) * np.power(yd_val, 2))
                       + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                       + np.power(ye_val, 4)))
               - (3 * np.power(ye_val, 2) * ((3 * (np.power(yb_val, 2)
                                                   + np.power(ys_val, 2)
                                                   + np.power(yd_val, 2)))
                                             + np.power(ytau_val, 2)
                                             + np.power(ymu_val, 2)
                                             + np.power(ye_val, 2)))
               - (4 * np.power(yte_val, 4))
               + (((16 * np.power(g3_val, 2))
                   - ((2 / 5) * np.power(g1_val, 2)))
                  * (np.power(yb_val, 2) + np.power(ys_val, 2)
                     + np.power(yd_val, 2)))
               + ((6 / 5) * np.power(g1_val, 2) * (np.power(ytau_val, 2)
                                                   + np.power(ymu_val, 2)
                                                   + np.power(ye_val, 2)))
               + (6 * np.power(g2_val, 2) * np.power(ye_val, 2))
               + ((15 / 2) * np.power(g2_val, 4))
               + ((9 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2))
               + ((27 / 2) * np.power(g1_val, 4)))

        # Total Yukawa coupling beta functions
        dyt_dt = (1 / t) * ((loop_fac * dyt_dt_1l)
                            + (loop_fac_sq * dyt_dt_2l))

        dyc_dt = (1 / t) * ((loop_fac * dyc_dt_1l)
                            + (loop_fac_sq * dyc_dt_2l))

        dyu_dt = (1 / t) * ((loop_fac * dyu_dt_1l)
                            + (loop_fac_sq * dyu_dt_2l))

        dyb_dt = (1 / t) * ((loop_fac * dyb_dt_1l)
                            + (loop_fac_sq * dyb_dt_2l))

        dys_dt = (1 / t) * ((loop_fac * dys_dt_1l)
                            + (loop_fac_sq * dys_dt_2l))

        dyd_dt = (1 / t) * ((loop_fac * dyd_dt_1l)
                            + (loop_fac_sq * dyd_dt_2l))

        dytau_dt = (1 / t) * ((loop_fac * dytau_dt_1l)
                            + (loop_fac_sq * dytau_dt_2l))

        dymu_dt = (1 / t) * ((loop_fac * dymu_dt_1l)
                            + (loop_fac_sq * dymu_dt_2l))

        dye_dt = (1 / t) * ((loop_fac * dye_dt_1l)
                            + (loop_fac_sq * dye_dt_2l))

        ##### Soft trilinear couplings for 3 gen, assumed diagonalized #####
        # 1 loop parts
        dat_dt_1l = (at_val
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        + (5 * np.power(yt_val, 2)) + np.power(yb_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))\
            + (yt_val
               * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                        + (au_val * yu_val))) + (4 * yt_val * at_val)
                  + (2 * yb_val * ab_val)
                  + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                  + (6 * np.power(g2_val, 2) * M2_val)
                  + ((26 / 15) * np.power(g1_val, 2) * M1_val)))

        dac_dt_1l = (ac_val
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        + (5 * np.power(yc_val, 2)) + np.power(ys_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))\
            + (yc_val
               * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                        + (au_val * yu_val))) + (4 * yc_val * ac_val)
                  + (2 * ys_val * as_val)
                  + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                  + (6 * np.power(g2_val, 2) * M2_val)
                  + ((26 / 15) * np.power(g1_val, 2) * M1_val)))

        dau_dt_1l = (au_val
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        + (5 * np.power(yu_val, 2)) + np.power(yd_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))\
            + (yu_val
               * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                        + (au_val * yu_val))) + (4 * yu_val * au_val)
                  + (2 * yd_val * ad_val)
                  + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                  + (6 * np.power(g2_val, 2) * M2_val)
                  + ((26 / 15) * np.power(g1_val, 2) * M1_val)))

        dab_dt_1l = (ab_val
                     * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                         + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                            + np.power(ye_val, 2)))
                        + (5 * np.power(yb_val, 2)) + np.power(yt_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))\
            + (yb_val
               * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                        + (ad_val * yd_val)))
                  + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                          + (ae_val * ye_val)))
                  + (4 * yb_val * ab_val) + (2 * yt_val * at_val)
                  + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                  + (6 * np.power(g2_val, 2) * M2_val)
                  + ((14 / 15) * np.power(g1_val, 2) * M1_val)))

        das_dt_1l = (as_val
                     * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                         + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                            + np.power(ye_val, 2)))
                        + (5 * np.power(ys_val, 2)) + np.power(yc_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))\
            + (ys_val
               * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                        + (ad_val * yd_val)))
                  + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                          + (ae_val * ye_val)))
                  + (4 * ys_val * as_val) + (2 * yc_val * ac_val)
                  + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                  + (6 * np.power(g2_val, 2) * M2_val)
                  + ((14 / 15) * np.power(g1_val, 2) * M1_val)))

        dad_dt_1l = (ad_val
                     * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                         + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                            + np.power(ye_val, 2)))
                        + (5 * np.power(yd_val, 2)) + np.power(yu_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))\
            + (yd_val
               * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                        + (ad_val * yd_val)))
                  + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                          + (ae_val * ye_val)))
                  + (4 * yd_val * ad_val) + (2 * yu_val * au_val)
                  + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                  + (6 * np.power(g2_val, 2) * M2_val)
                  + ((14 / 15) * np.power(g1_val, 2) * M1_val)))

        datau_dt_1l = (atau_val
                     * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                         + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                            + np.power(ye_val, 2)))
                        + (5 * np.power(ytau_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((9 / 5) * np.power(g1_val, 2))))\
            + (ytau_val
               * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                        + (ad_val * yd_val)))
                  + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                          + (ae_val * ye_val)))
                  + (4 * ytau_val * atau_val)
                  + (6 * np.power(g2_val, 2) * M2_val)
                  + ((18 / 5) * np.power(g1_val, 2) * M1_val)))

        damu_dt_1l = (amu_val
                     * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                         + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                            + np.power(ye_val, 2)))
                        + (5 * np.power(ymu_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((9 / 5) * np.power(g1_val, 2))))\
            + (ymu_val
               * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                        + (ad_val * yd_val)))
                  + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                          + (ae_val * ye_val)))
                  + (4 * ymu_val * amu_val)
                  + (6 * np.power(g2_val, 2) * M2_val)
                  + ((18 / 5) * np.power(g1_val, 2) * M1_val)))

        dae_dt_1l = (ae_val
                     * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                         + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                            + np.power(ye_val, 2)))
                        + (5 * np.power(ye_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((9 / 5) * np.power(g1_val, 2))))\
            + (ye_val
               * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                        + (ad_val * yd_val)))
                  + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                          + (ae_val * ye_val)))
                  + (4 * ye_val * ae_val)
                  + (6 * np.power(g2_val, 2) * M2_val)
                  + ((18 / 5) * np.power(g1_val, 2) * M1_val)))

        # 2 loop parts
        dat_dt_2l = (at_val
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                    + (np.power(yc_val, 2)
                                       * np.power(ys_val, 2))
                                    + (np.power(yu_val, 2)
                                       * np.power(yd_val, 2)))))
                        - (np.power(yb_val, 2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))
                        - (15 * np.power(yt_val, 2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        - (6 * np.power(yt_val, 4))
                        - (2 * np.power(yb_val, 4))
                        - (4 * np.power(yb_val, 2) * np.power(yt_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        + (12 * np.power(g2_val, 2)
                           * np.power(yt_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2)
                           * np.power(yb_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2)
                           * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))\
            + (yt_val
               * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                 + (ac_val * np.power(yc_val, 3))
                                 + (au_val * np.power(yu_val, 3))))
                           + (at_val * np.power(yb_val, 2) * yt_val)
                           + (ac_val * np.power(ys_val, 2) * yc_val)
                           + (au_val * np.power(yd_val, 2) * yu_val)
                           + (ab_val * np.power(yt_val, 2) * yb_val)
                           + (as_val * np.power(yc_val, 2) * ys_val)
                           + (ad_val * np.power(yu_val, 2) * yd_val)))
                  - (18 * np.power(yt_val, 2)
                     * ((at_val * yt_val)
                        + (ac_val * yc_val)
                        + (au_val * yu_val)))
                  - (np.power(yb_val, 2)
                     * ((6
                         * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + (2
                           * ((atau_val * ytau_val) + (amu_val * ymu_val)
                              + (ae_val * ye_val)))))
                  - (12 * yt_val * at_val
                     * (np.power(yt_val, 2) + np.power(yc_val, 2)
                        + np.power(yu_val, 2)))
                  - (yb_val * ab_val
                     * ((6 * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (2 * (np.power(ytau_val, 2)
                                + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))))
                  - (14 * np.power(yt_val, 3) * at_val)
                  - (8 * np.power(yb_val, 3) * ab_val)
                  - (2 * np.power(yb_val, 2) * yt_val * at_val)
                  - (4 * yb_val * ab_val * np.power(yt_val, 2))
                  + (((32 * np.power(g3_val, 2))
                      + ((8 / 5) * np.power(g1_val, 2)))
                     * ((at_val * yt_val) + (ac_val * yc_val)
                        + (au_val * yu_val)))
                  + (((6 * np.power(g2_val, 2))
                      + ((6 / 5) * np.power(g1_val, 2)))
                     * yt_val * at_val)
                  + ((4 / 5) * np.power(g1_val, 2) * yb_val * ab_val)
                  - (((32 * np.power(g3_val, 2) * M3_val)
                      + (8 / 5) * np.power(g1_val, 2) * M1_val)
                     * (np.power(yt_val, 2) + np.power(yc_val, 2)
                        + np.power(yu_val, 2)))
                  - (((12 * np.power(g2_val, 2) * M2_val)
                      + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                     * np.power(yt_val, 2))
                  - ((4 / 5) * np.power(g1_val, 2) * M1_val
                     * np.power(yb_val, 2))
                  + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                  - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                     * (M3_val + M2_val))
                  - ((272 / 45) * np.power(g3_val, 2) * np.power(g1_val, 2)
                     * (M3_val + M1_val))
                  - (30 * np.power(g2_val, 4) * M2_val)
                  - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                     * (M2_val + M1_val))
                  - ((5486 / 225) * np.power(g1_val, 4) * M1_val)))

        dac_dt_2l = (ac_val
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                    + (np.power(yc_val, 2)
                                       * np.power(ys_val, 2))
                                    + (np.power(yu_val, 2)
                                       * np.power(yd_val, 2)))))
                        - (np.power(ys_val, 2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))
                        - (15 * np.power(yc_val, 2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        - (6 * np.power(yc_val, 4))
                        - (2 * np.power(ys_val, 4))
                        - (4 * np.power(ys_val, 2) * np.power(yc_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        + (12 * np.power(g2_val, 2)
                           * np.power(yc_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2)
                           * np.power(ys_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2)
                           * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))\
            + (yc_val
               * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                 + (ac_val * np.power(yc_val, 3))
                                 + (au_val * np.power(yu_val, 3))))
                           + (at_val * np.power(yb_val, 2) * yt_val)
                           + (ac_val * np.power(ys_val, 2) * yc_val)
                           + (au_val * np.power(yd_val, 2) * yu_val)
                           + (ab_val * np.power(yt_val, 2) * yb_val)
                           + (as_val * np.power(yc_val, 2) * ys_val)
                           + (ad_val * np.power(yu_val, 2) * yd_val)))
                  - (18 * np.power(yc_val, 2)
                     * ((at_val * yt_val)
                        + (ac_val * yc_val)
                        + (au_val * yu_val)))
                  - (np.power(ys_val, 2)
                     * ((6
                         * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + (2
                           * ((atau_val * ytau_val) + (amu_val * ymu_val)
                              + (ae_val * ye_val)))))
                  - (12 * yc_val * ac_val
                     * (np.power(yt_val, 2) + np.power(yc_val, 2)
                        + np.power(yu_val, 2)))
                  - (ys_val * as_val
                     * ((6 * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (2 * (np.power(ytau_val, 2)
                                + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))))
                  - (14 * np.power(yc_val, 3) * ac_val)
                  - (8 * np.power(ys_val, 3) * as_val)
                  - (2 * np.power(ys_val, 2) * yc_val * ac_val)
                  - (4 * ys_val * as_val * np.power(yc_val, 2))
                  + (((32 * np.power(g3_val, 2))
                      + ((8 / 5) * np.power(g1_val, 2)))
                     * ((at_val * yt_val) + (ac_val * yc_val)
                        + (au_val * yu_val)))
                  + (((6 * np.power(g2_val, 2))
                      + ((6 / 5) * np.power(g1_val, 2)))
                     * yc_val * ac_val)
                  + ((4 / 5) * np.power(g1_val, 2) * ys_val * as_val)
                  - (((32 * np.power(g3_val, 2) * M3_val)
                      + (8 / 5) * np.power(g1_val, 2) * M1_val)
                     * (np.power(yt_val, 2) + np.power(yc_val, 2)
                        + np.power(yu_val, 2)))
                  - (((12 * np.power(g2_val, 2) * M2_val)
                      + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                     * np.power(yc_val, 2))
                  - ((4 / 5) * np.power(g1_val, 2) * M1_val
                     * np.power(ys_val, 2))
                  + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                  - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                     * (M3_val + M2_val))
                  - ((272 / 45) * np.power(g3_val, 2) * np.power(g1_val, 2)
                     * (M3_val + M1_val))
                  - (30 * np.power(g2_val, 4) * M2_val)
                  - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                     * (M2_val + M1_val))
                  - ((5486 / 225) * np.power(g1_val, 4) * M1_val)))

        dau_dt_2l = (au_val
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                    + (np.power(yc_val, 2)
                                       * np.power(ys_val, 2))
                                    + (np.power(yu_val, 2)
                                       * np.power(yd_val, 2)))))
                        - (np.power(yd_val, 2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))
                        - (15 * np.power(yu_val, 2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        - (6 * np.power(yu_val, 4))
                        - (2 * np.power(yd_val, 4))
                        - (4 * np.power(yd_val, 2) * np.power(yu_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        + (12 * np.power(g2_val, 2)
                           * np.power(yu_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2)
                           * np.power(yd_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2)
                           * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))\
            + (yu_val
               * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                 + (ac_val * np.power(yc_val, 3))
                                 + (au_val * np.power(yu_val, 3))))
                           + (at_val * np.power(yb_val, 2) * yt_val)
                           + (ac_val * np.power(ys_val, 2) * yc_val)
                           + (au_val * np.power(yd_val, 2) * yu_val)
                           + (ab_val * np.power(yt_val, 2) * yb_val)
                           + (as_val * np.power(yc_val, 2) * ys_val)
                           + (ad_val * np.power(yu_val, 2) * yd_val)))
                  - (18 * np.power(yu_val, 2)
                     * ((at_val * yt_val)
                        + (ac_val * yc_val)
                        + (au_val * yu_val)))
                  - (np.power(yd_val, 2)
                     * ((6
                         * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + (2
                           * ((atau_val * ytau_val) + (amu_val * ymu_val)
                              + (ae_val * ye_val)))))
                  - (12 * yu_val * au_val
                     * (np.power(yt_val, 2) + np.power(yc_val, 2)
                        + np.power(yu_val, 2)))
                  - (yd_val * ad_val
                     * ((6 * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (2 * (np.power(ytau_val, 2)
                                + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))))
                  - (14 * np.power(yu_val, 3) * au_val)
                  - (8 * np.power(yd_val, 3) * ad_val)
                  - (2 * np.power(yd_val, 2) * yu_val * au_val)
                  - (4 * yd_val * ad_val * np.power(yu_val, 2))
                  + (((32 * np.power(g3_val, 2))
                      + ((8 / 5) * np.power(g1_val, 2)))
                     * ((at_val * yt_val) + (ac_val * yc_val)
                        + (au_val * yu_val)))
                  + (((6 * np.power(g2_val, 2))
                      + ((6 / 5) * np.power(g1_val, 2)))
                     * yu_val * au_val)
                  + ((4 / 5) * np.power(g1_val, 2) * yd_val * ad_val)
                  - (((32 * np.power(g3_val, 2) * M3_val)
                      + (8 / 5) * np.power(g1_val, 2) * M1_val)
                     * (np.power(yt_val, 2) + np.power(yc_val, 2)
                        + np.power(yu_val, 2)))
                  - (((12 * np.power(g2_val, 2) * M2_val)
                      + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                     * np.power(yu_val, 2))
                  - ((4 / 5) * np.power(g1_val, 2) * M1_val
                     * np.power(yd_val, 2))
                  + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                  - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                     * (M3_val + M2_val))
                  - ((272 / 45) * np.power(g3_val, 2) * np.power(g1_val, 2)
                     * (M3_val + M1_val))
                  - (30 * np.power(g2_val, 4) * M2_val)
                  - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                     * (M2_val + M1_val))
                  - ((5486 / 225) * np.power(g1_val, 4) * M1_val)))

        dab_dt_2l = (ab_val # Tr(3Yd^4+Yu^2*Yd^2+Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                    + (np.power(yc_val, 2)
                                       * np.power(ys_val, 2))
                                    + (np.power(yu_val, 2)
                                       * np.power(yd_val, 2)))
                                 + np.power(ytau_val, 4)
                                 + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))
                        - (3 * np.power(yt_val, 2) # Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        - (5 * np.power(yb_val, 2) # Tr(3Yd^2+Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))
                        - (6 * np.power(yb_val, 4))
                        - (2 * np.power(yt_val, 4))
                        - (4 * np.power(yb_val, 2) * np.power(yt_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2))) # Tr(Yd^2)
                           * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + ((6 / 5) * np.power(g1_val, 2) # Tr(Ye^2)
                           * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))
                        + ((4 / 5) * np.power(g1_val, 2)
                           * np.power(yt_val, 2))
                        + (((12 * np.power(g2_val, 2))
                            + ((6 / 5) * np.power(g1_val, 2)))
                           * np.power(yb_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2)
                           * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))\
            + (yb_val # Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
               * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                 + (as_val * np.power(ys_val, 3))
                                 + (ad_val * np.power(yd_val, 3))))
                           + (at_val * np.power(yb_val, 2) * yt_val)
                           + (ac_val * np.power(ys_val, 2) * yc_val)
                           + (au_val * np.power(yd_val, 2) * yu_val)
                           + (ab_val * np.power(yt_val, 2) * yb_val)
                           + (as_val * np.power(yc_val, 2) * ys_val)
                           + (ad_val * np.power(yu_val, 2) * yd_val)
                           + (2 * ((atau_val * np.power(ytau_val, 3))
                                   + (amu_val * np.power(ymu_val, 3))
                                   + (ae_val * np.power(ye_val, 3))))))
                  - (6 * np.power(yt_val, 2) # Tr(au*Yu)
                     * ((at_val * yt_val)
                        + (ac_val * yc_val)
                        + (au_val * yu_val)))
                  - (6 * np.power(yb_val, 2) # Tr(3ad*Yd + ae*Ye)
                     * ((3
                         * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + ((atau_val * ytau_val) + (amu_val * ymu_val)
                           + (ae_val * ye_val))))
                  - (6 * yt_val * at_val # Tr(Yu^2)
                     * (np.power(yt_val, 2) + np.power(yc_val, 2)
                        + np.power(yu_val, 2)))
                  - (4 * yb_val * ab_val # Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + ((np.power(ytau_val, 2)
                            + np.power(ymu_val, 2)
                            + np.power(ye_val, 2)))))
                  - (14 * np.power(yb_val, 3) * ab_val)
                  - (8 * np.power(yt_val, 3) * at_val)
                  - (4 * np.power(yb_val, 2) * yt_val * at_val)
                  - (2 * yb_val * ab_val * np.power(yt_val, 2))
                  + (((32 * np.power(g3_val, 2))
                      - ((4 / 5) * np.power(g1_val, 2))) # Tr(ad*Yd)
                     * ((ab_val * yb_val) + (as_val * ys_val)
                        + (ad_val * yd_val)))
                  + ((12 / 5) * np.power(g1_val, 2) # Tr(ae*Ye)
                     * ((atau_val * ytau_val) + (amu_val * ymu_val)
                        + (ae_val * ye_val)))
                  + ((8 / 5) * np.power(g1_val, 2) * yt_val * at_val)
                  + (((6 * np.power(g2_val, 2))
                      + ((6 / 5) * np.power(g1_val, 2)))
                     * yb_val * ab_val)
                  - (((32 * np.power(g3_val, 2) * M3_val)
                      - (4 / 5) * np.power(g1_val, 2) * M1_val) # Tr(Yd^2)
                     * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  - ((12 / 5) * np.power(g1_val, 2) * M1_val # Tr(Ye^2)
                     * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                        + np.power(ye_val, 2)))
                  - (((12 * np.power(g2_val, 2) * M2_val)
                      + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                     * np.power(yb_val, 2))
                  - ((8 / 5) * np.power(g1_val, 2) * M1_val
                     * np.power(yt_val, 2))
                  + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                  - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                     * (M3_val + M2_val))
                  - ((16 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2)
                     * (M3_val + M1_val))
                  - (30 * np.power(g2_val, 4) * M2_val)
                  - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                     * (M2_val + M1_val))
                  - ((574 / 45) * np.power(g1_val, 4) * M1_val)))

        das_dt_2l = (as_val # Tr(3Yd^4+Yu^2*Yd^2+Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                    + (np.power(yc_val, 2)
                                       * np.power(ys_val, 2))
                                    + (np.power(yu_val, 2)
                                       * np.power(yd_val, 2)))
                                 + np.power(ytau_val, 4)
                                 + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))
                        - (3 * np.power(yc_val, 2) # Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        - (5 * np.power(ys_val, 2) # Tr(3Yd^2+Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))
                        - (6 * np.power(ys_val, 4))
                        - (2 * np.power(yc_val, 4))
                        - (4 * np.power(ys_val, 2) * np.power(yc_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2))) # Tr(Yd^2)
                           * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + ((6 / 5) * np.power(g1_val, 2) # Tr(Ye^2)
                           * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))
                        + ((4 / 5) * np.power(g1_val, 2)
                           * np.power(yc_val, 2))
                        + (((12 * np.power(g2_val, 2))
                            + ((6 / 5) * np.power(g1_val, 2)))
                           * np.power(ys_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2)
                           * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))\
            + (ys_val # Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
               * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                 + (as_val * np.power(ys_val, 3))
                                 + (ad_val * np.power(yd_val, 3))))
                           + (at_val * np.power(yb_val, 2) * yt_val)
                           + (ac_val * np.power(ys_val, 2) * yc_val)
                           + (au_val * np.power(yd_val, 2) * yu_val)
                           + (ab_val * np.power(yt_val, 2) * yb_val)
                           + (as_val * np.power(yc_val, 2) * ys_val)
                           + (ad_val * np.power(yu_val, 2) * yd_val)
                           + (2 * ((atau_val * np.power(ytau_val, 3))
                                   + (amu_val * np.power(ymu_val, 3))
                                   + (ae_val * np.power(ye_val, 3))))))
                  - (6 * np.power(yc_val, 2) # Tr(au*Yu)
                     * ((at_val * yt_val)
                        + (ac_val * yc_val)
                        + (au_val * yu_val)))
                  - (6 * np.power(ys_val, 2) # Tr(3ad*Yd + ae*Ye)
                     * ((3
                         * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + ((atau_val * ytau_val) + (amu_val * ymu_val)
                           + (ae_val * ye_val))))
                  - (6 * yc_val * ac_val # Tr(Yu^2)
                     * (np.power(yt_val, 2) + np.power(yc_val, 2)
                        + np.power(yu_val, 2)))
                  - (4 * ys_val * as_val # Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + ((np.power(ytau_val, 2)
                            + np.power(ymu_val, 2)
                            + np.power(ye_val, 2)))))
                  - (14 * np.power(ys_val, 3) * as_val)
                  - (8 * np.power(yc_val, 3) * ac_val)
                  - (4 * np.power(ys_val, 2) * yc_val * ac_val)
                  - (2 * ys_val * as_val * np.power(yc_val, 2))
                  + (((32 * np.power(g3_val, 2))
                      - ((4 / 5) * np.power(g1_val, 2))) # Tr(ad*Yd)
                     * ((ab_val * yb_val) + (as_val * ys_val)
                        + (ad_val * yd_val)))
                  + ((12 / 5) * np.power(g1_val, 2) # Tr(ae*Ye)
                     * ((atau_val * ytau_val) + (amu_val * ymu_val)
                        + (ae_val * ye_val)))
                  + ((8 / 5) * np.power(g1_val, 2) * yc_val * ac_val)
                  + (((6 * np.power(g2_val, 2))
                      + ((6 / 5) * np.power(g1_val, 2)))
                     * ys_val * as_val)
                  - (((32 * np.power(g3_val, 2) * M3_val)
                      - (4 / 5) * np.power(g1_val, 2) * M1_val) # Tr(Yd^2)
                     * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  - ((12 / 5) * np.power(g1_val, 2) * M1_val # Tr(Ye^2)
                     * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                        + np.power(ye_val, 2)))
                  - (((12 * np.power(g2_val, 2) * M2_val)
                      + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                     * np.power(ys_val, 2))
                  - ((8 / 5) * np.power(g1_val, 2) * M1_val
                     * np.power(yc_val, 2))
                  + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                  - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                     * (M3_val + M2_val))
                  - ((16 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2)
                     * (M3_val + M1_val))
                  - (30 * np.power(g2_val, 4) * M2_val)
                  - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                     * (M2_val + M1_val))
                  - ((574 / 45) * np.power(g1_val, 4) * M1_val)))

        dad_dt_2l = (ad_val # Tr(3Yd^4+Yu^2*Yd^2+Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                    + (np.power(yc_val, 2)
                                       * np.power(ys_val, 2))
                                    + (np.power(yu_val, 2)
                                       * np.power(yd_val, 2)))
                                 + np.power(ytau_val, 4)
                                 + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))
                        - (3 * np.power(yu_val, 2) # Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        - (5 * np.power(yd_val, 2) # Tr(3Yd^2+Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))
                        - (6 * np.power(yd_val, 4))
                        - (2 * np.power(yu_val, 4))
                        - (4 * np.power(yd_val, 2) * np.power(yu_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2))) # Tr(Yd^2)
                           * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + ((6 / 5) * np.power(g1_val, 2) # Tr(Ye^2)
                           * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))
                        + ((4 / 5) * np.power(g1_val, 2)
                           * np.power(yu_val, 2))
                        + (((12 * np.power(g2_val, 2))
                            + ((6 / 5) * np.power(g1_val, 2)))
                           * np.power(yd_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2)
                           * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))\
            + (yd_val # Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
               * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                 + (as_val * np.power(ys_val, 3))
                                 + (ad_val * np.power(yd_val, 3))))
                           + (at_val * np.power(yb_val, 2) * yt_val)
                           + (ac_val * np.power(ys_val, 2) * yc_val)
                           + (au_val * np.power(yd_val, 2) * yu_val)
                           + (ab_val * np.power(yt_val, 2) * yb_val)
                           + (as_val * np.power(yc_val, 2) * ys_val)
                           + (ad_val * np.power(yu_val, 2) * yd_val)
                           + (2 * ((atau_val * np.power(ytau_val, 3))
                                   + (amu_val * np.power(ymu_val, 3))
                                   + (ae_val * np.power(ye_val, 3))))))
                  - (6 * np.power(yu_val, 2) # Tr(au*Yu)
                     * ((at_val * yt_val)
                        + (ac_val * yc_val)
                        + (au_val * yu_val)))
                  - (6 * np.power(yd_val, 2) # Tr(3ad*Yd + ae*Ye)
                     * ((3
                         * ((ab_val * yb_val) + (as_val * ys_val)
                            + (ad_val * yd_val)))
                        + ((atau_val * ytau_val) + (amu_val * ymu_val)
                           + (ae_val * ye_val))))
                  - (6 * yu_val * au_val # Tr(Yu^2)
                     * (np.power(yt_val, 2) + np.power(yc_val, 2)
                        + np.power(yu_val, 2)))
                  - (4 * yd_val * ad_val # Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + ((np.power(ytau_val, 2)
                            + np.power(ymu_val, 2)
                            + np.power(ye_val, 2)))))
                  - (14 * np.power(yd_val, 3) * ad_val)
                  - (8 * np.power(yu_val, 3) * au_val)
                  - (4 * np.power(yd_val, 2) * yu_val * au_val)
                  - (2 * yd_val * ad_val * np.power(yu_val, 2))
                  + (((32 * np.power(g3_val, 2))
                      - ((4 / 5) * np.power(g1_val, 2))) # Tr(ad*Yd)
                     * ((ab_val * yb_val) + (as_val * ys_val)
                        + (ad_val * yd_val)))
                  + ((12 / 5) * np.power(g1_val, 2) # Tr(ae*Ye)
                     * ((atau_val * ytau_val) + (amu_val * ymu_val)
                        + (ae_val * ye_val)))
                  + ((8 / 5) * np.power(g1_val, 2) * yu_val * au_val)
                  + (((6 * np.power(g2_val, 2))
                      + ((6 / 5) * np.power(g1_val, 2)))
                     * yd_val * ad_val)
                  - (((32 * np.power(g3_val, 2) * M3_val)
                      - (4 / 5) * np.power(g1_val, 2) * M1_val) # Tr(Yd^2)
                     * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  - ((12 / 5) * np.power(g1_val, 2) * M1_val # Tr(Ye^2)
                     * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                        + np.power(ye_val, 2)))
                  - (((12 * np.power(g2_val, 2) * M2_val)
                      + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                     * np.power(yd_val, 2))
                  - ((8 / 5) * np.power(g1_val, 2) * M1_val
                     * np.power(yu_val, 2))
                  + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                  - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                     * (M3_val + M2_val))
                  - ((16 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2)
                     * (M3_val + M1_val))
                  - (30 * np.power(g2_val, 4) * M2_val)
                  - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                     * (M2_val + M1_val))
                  - ((574 / 45) * np.power(g1_val, 4) * M1_val)))

        datau_dt_2l = (atau_val # Tr(3Yd^4+Yu^2*Yd^2+Ye^4)
                       * (((-3) * ((3 * (np.power(yb_val, 4)
                                         + np.power(ys_val, 4)
                                         + np.power(yd_val, 4)))
                                   + ((np.power(yt_val, 2)
                                       * np.power(yb_val,2))
                                      + (np.power(yc_val, 2)
                                         * np.power(ys_val, 2))
                                      + (np.power(yu_val, 2)
                                         * np.power(yd_val, 2)))
                                   + np.power(ytau_val, 4)
                                   + np.power(ymu_val, 4)
                                   + np.power(ye_val, 4))) # end trace
                          - (5 * np.power(ytau_val, 2) # Tr(3Yd^2+Ye^2)
                             * ((3 * (np.power(yb_val, 2)
                                      + np.power(ys_val, 2)
                                      + np.power(yd_val, 2)))
                                + (np.power(ytau_val, 2)
                                   + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))) # end trace
                          - (6 * np.power(ytau_val, 4))
                          + (((16 * np.power(g3_val, 2))
                              - ((2 / 5) * np.power(g1_val, 2))) # Tr(Yd^2)
                             * (np.power(yb_val, 2)
                                + np.power(ys_val, 2)
                                + np.power(yd_val, 2))) # end trace
                          + ((6 / 5) * np.power(g1_val, 2) # Tr(Ye^2)
                             * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                + np.power(ye_val, 2))) # end trace
                          + (((12 * np.power(g2_val, 2))
                              - ((6 / 5) * np.power(g1_val, 2)))
                             * np.power(ytau_val, 2))
                          + ((15 / 2) * np.power(g2_val, 4))
                          + ((9 / 5) * np.power(g2_val, 2)
                             * np.power(g1_val, 2))
                          + ((27 / 2) * np.power(g1_val, 4))))\
            + (ytau_val # Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
               * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                 + (as_val * np.power(ys_val, 3))
                                 + (ad_val * np.power(yd_val, 3))))
                           + (at_val * np.power(yb_val, 2) * yt_val)
                           + (ac_val * np.power(ys_val, 2) * yc_val)
                           + (au_val * np.power(yd_val, 2) * yu_val)
                           + (ab_val * np.power(yt_val, 2) * yb_val)
                           + (as_val * np.power(yc_val, 2) * ys_val)
                           + (ad_val * np.power(yu_val, 2) * yd_val)
                           + (2 * ((atau_val * np.power(ytau_val, 3))
                                   + (amu_val * np.power(ymu_val, 3))
                                   + (ae_val * np.power(ye_val, 3)))))) # end trace
                  - (4 * ytau_val * atau_val # Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + ((np.power(ytau_val, 2)
                            + np.power(ymu_val, 2)
                            + np.power(ye_val, 2))))) # end trace
                  - (6 * np.power(ytau_val, 2) # Tr(3ad*Yd + ae*Ye)
                     * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                              + (ad_val * yd_val)))
                        + (atau_val * ytau_val) + (amu_val * ymu_val)
                        + (ae_val * ye_val))) # end trace
                  - (14 * np.power(ytau_val, 3) * atau_val)
                  + (((32 * np.power(g3_val, 2))
                      - ((4 / 5) * np.power(g1_val, 2))) # Tr(ad*Yd)
                     * ((ab_val * yb_val) + (as_val * ys_val)
                        + (ad_val * yd_val))) # end trace
                  + ((12 / 5) * np.power(g1_val, 2) # Tr(ae*Ye)
                     * ((atau_val * ytau_val) + (amu_val * ymu_val)
                        + (ae_val * ye_val))) # end trace
                  + (((6 * np.power(g2_val, 2))
                      + ((6 / 5) * np.power(g1_val, 2)))
                         * ytau_val * atau_val)
                  - (((32 * np.power(g3_val, 2) * M3_val)
                      - (4 / 5) * np.power(g1_val, 2) * M1_val) # Tr(Yd^2)
                     * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2))) # end trace
                  - ((12 / 5) * np.power(g1_val, 2) * M1_val # Tr(Ye^2)
                     * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                        + np.power(ye_val, 2))) # end trace
                  - (12 * np.power(g2_val, 2) * M2_val
                     * np.power(ytau_val, 2))
                  - (30 * np.power(g2_val, 4) * M2_val)
                  - ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                     * (M1_val + M2_val))
                  - (54 * np.power(g1_val, 4) * M1_val)))

        damu_dt_2l = (amu_val # Tr(3Yd^4+Yu^2*Yd^2+Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + ((np.power(yt_val, 2)
                                      * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4))) # end trace
                         - (5 * np.power(ymu_val, 2) # Tr(3Yd^2+Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2)))) # end trace
                         - (6 * np.power(ymu_val, 4))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2))) # Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2))) # end trace
                         + ((6 / 5) * np.power(g1_val, 2) # Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2))) # end trace
                         + (((12 * np.power(g2_val, 2))
                             - ((6 / 5) * np.power(g1_val, 2)))
                            * np.power(ymu_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + ((9 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2))
                         + ((27 / 2) * np.power(g1_val, 4))))\
            + (ymu_val # Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
               * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                 + (as_val * np.power(ys_val, 3))
                                 + (ad_val * np.power(yd_val, 3))))
                           + (at_val * np.power(yb_val, 2) * yt_val)
                           + (ac_val * np.power(ys_val, 2) * yc_val)
                           + (au_val * np.power(yd_val, 2) * yu_val)
                           + (ab_val * np.power(yt_val, 2) * yb_val)
                           + (as_val * np.power(yc_val, 2) * ys_val)
                           + (ad_val * np.power(yu_val, 2) * yd_val)
                           + (2 * ((atau_val * np.power(ytau_val, 3))
                                   + (amu_val * np.power(ymu_val, 3))
                                   + (ae_val * np.power(ye_val, 3)))))) # end trace
                  - (4 * ymu_val * amu_val # Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + ((np.power(ytau_val, 2)
                            + np.power(ymu_val, 2)
                            + np.power(ye_val, 2))))) # end trace
                  - (6 * np.power(ymu_val, 2) # Tr(3ad*Yd + ae*Ye)
                     * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                              + (ad_val * yd_val)))
                        + (atau_val * ytau_val) + (amu_val * ymu_val)
                        + (ae_val * ye_val))) # end trace
                  - (14 * np.power(ymu_val, 3) * amu_val)
                  + (((32 * np.power(g3_val, 2))
                      - ((4 / 5) * np.power(g1_val, 2))) # Tr(ad*Yd)
                     * ((ab_val * yb_val) + (as_val * ys_val)
                        + (ad_val * yd_val))) # end trace
                  + ((12 / 5) * np.power(g1_val, 2) # Tr(ae*Ye)
                     * ((atau_val * ytau_val) + (amu_val * ymu_val)
                        + (ae_val * ye_val))) # end trace
                  + (((6 * np.power(g2_val, 2))
                      + ((6 / 5) * np.power(g1_val, 2)))
                         * ymu_val * amu_val)
                  - (((32 * np.power(g3_val, 2) * M3_val)
                      - (4 / 5) * np.power(g1_val, 2) * M1_val) # Tr(Yd^2)
                     * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2))) # end trace
                  - ((12 / 5) * np.power(g1_val, 2) * M1_val # Tr(Ye^2)
                     * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                        + np.power(ye_val, 2))) # end trace
                  - (12 * np.power(g2_val, 2) * M2_val
                     * np.power(ymu_val, 2))
                  - (30 * np.power(g2_val, 4) * M2_val)
                  - ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                     * (M1_val + M2_val))
                  - (54 * np.power(g1_val, 4) * M1_val)))

        dae_dt_2l = (ae_val # Tr(3Yd^4+Yu^2*Yd^2+Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + ((np.power(yt_val, 2)
                                     * np.power(yb_val,2))
                                    + (np.power(yc_val, 2)
                                       * np.power(ys_val, 2))
                                    + (np.power(yu_val, 2)
                                       * np.power(yd_val, 2)))
                                 + np.power(ytau_val, 4)
                                 + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4))) # end trace
                        - (5 * np.power(ye_val, 2) # Tr(3Yd^2+Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))) # end trace
                        - (6 * np.power(ye_val, 4))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2))) # Tr(Yd^2)
                           * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2))) # end trace
                        + ((6 / 5) * np.power(g1_val, 2) # Tr(Ye^2)
                           * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2))) # end trace
                        + (((12 * np.power(g2_val, 2))
                            - ((6 / 5) * np.power(g1_val, 2)))
                           * np.power(ye_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + ((9 / 5) * np.power(g2_val, 2)
                           * np.power(g1_val, 2))
                        + ((27 / 2) * np.power(g1_val, 4))))\
            + (ye_val # Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
               * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                 + (as_val * np.power(ys_val, 3))
                                 + (ad_val * np.power(yd_val, 3))))
                           + (at_val * np.power(yb_val, 2) * yt_val)
                           + (ac_val * np.power(ys_val, 2) * yc_val)
                           + (au_val * np.power(yd_val, 2) * yu_val)
                           + (ab_val * np.power(yt_val, 2) * yb_val)
                           + (as_val * np.power(yc_val, 2) * ys_val)
                           + (ad_val * np.power(yu_val, 2) * yd_val)
                           + (2 * ((atau_val * np.power(ytau_val, 3))
                                   + (amu_val * np.power(ymu_val, 3))
                                   + (ae_val * np.power(ye_val, 3)))))) # end trace
                  - (4 * ye_val * ae_val # Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2)
                              + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + ((np.power(ytau_val, 2)
                            + np.power(ymu_val, 2)
                            + np.power(ye_val, 2))))) # end trace
                  - (6 * np.power(ye_val, 2) # Tr(3ad*Yd + ae*Ye)
                     * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                              + (ad_val * yd_val)))
                        + (atau_val * ytau_val) + (amu_val * ymu_val)
                        + (ae_val * ye_val))) # end trace
                  - (14 * np.power(ye_val, 3) * ae_val)
                  + (((32 * np.power(g3_val, 2))
                      - ((4 / 5) * np.power(g1_val, 2))) # Tr(ad*Yd)
                     * ((ab_val * yb_val) + (as_val * ys_val)
                        + (ad_val * yd_val))) # end trace
                  + ((12 / 5) * np.power(g1_val, 2) # Tr(ae*Ye)
                     * ((atau_val * ytau_val) + (amu_val * ymu_val)
                        + (ae_val * ye_val))) # end trace
                  + (((6 * np.power(g2_val, 2))
                      + ((6 / 5) * np.power(g1_val, 2)))
                         * ye_val * ae_val)
                  - (((32 * np.power(g3_val, 2) * M3_val)
                      - (4 / 5) * np.power(g1_val, 2) * M1_val) # Tr(Yd^2)
                     * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2))) # end trace
                  - ((12 / 5) * np.power(g1_val, 2) * M1_val # Tr(Ye^2)
                     * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                        + np.power(ye_val, 2))) # end trace
                  - (12 * np.power(g2_val, 2) * M2_val
                     * np.power(ye_val, 2))
                  - (30 * np.power(g2_val, 4) * M2_val)
                  - ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                     * (M1_val + M2_val))
                  - (54 * np.power(g1_val, 4) * M1_val)))

        # Total soft trilinear coupling beta functions
        dat_dt = (1 / t) * ((loop_fac * dat_dt_1l)
                            + (loop_fac_sq * dat_dt_2l))

        dac_dt = (1 / t) * ((loop_fac * dac_dt_1l)
                            + (loop_fac_sq * dac_dt_2l))

        dau_dt = (1 / t) * ((loop_fac * dau_dt_1l)
                            + (loop_fac_sq * dau_dt_2l))

        dab_dt = (1 / t) * ((loop_fac * dab_dt_1l)
                            + (loop_fac_sq * dab_dt_2l))

        das_dt = (1 / t) * ((loop_fac * das_dt_1l)
                            + (loop_fac_sq * das_dt_2l))

        dad_dt = (1 / t) * ((loop_fac * dad_dt_1l)
                            + (loop_fac_sq * dad_dt_2l))

        datau_dt = (1 / t) * ((loop_fac * datau_dt_1l)
                            + (loop_fac_sq * datau_dt_2l))

        damu_dt = (1 / t) * ((loop_fac * damu_dt_1l)
                            + (loop_fac_sq * damu_dt_2l))

        dae_dt = (1 / t) * ((loop_fac * dae_dt_1l)
                            + (loop_fac_sq * dae_dt_2l))

        ##### Soft bilinear coupling b=B*mu#####
        # 1 loop part
        db_dt_1l = (b_val # Tr(3Yu^2 + 3Yd^2 + Ye^2)
                    * (((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2) + np.power(yb_val, 2)
                              + np.power(ys_val, 2) + np.power(yd_val, 2)))
                        + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                        + np.power(ye_val, 2)) # end trace
                       - (3 * np.power(g2_val, 2))
                       - ((3 / 5) * np.power(g1_val, 2))))\
            + (mu_val # Tr(6au*Yu + 6ad*Yd + 2ae*Ye)
               * (((6 * ((at_val * yt_val) + (ac_val * yc_val)
                         + (au_val * yu_val) + (ab_val * yb_val)
                         + (as_val * ys_val) + (ad_val * yd_val)))
                   + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                           + (ae_val * ye_val))))
                  + (6 * np.power(g2_val, 2) * M2_val)
                  + ((6 / 5) * np.power(g1_val, 2) * M1_val)))

        # 2 loop part
        db_dt_2l = (b_val # Tr(3Yu^4 + 3Yd^4 + 2Yu^2*Yd^2 + Ye^4)
                    * (((-3) * ((3 * (np.power(yt_val, 4) + np.power(yc_val, 4)
                                      + np.power(yu_val, 4)
                                      + np.power(yb_val, 4)
                                      + np.power(ys_val, 4)
                                      + np.power(yd_val, 4)))
                                + (2 * ((np.power(yt_val, 2)
                                         * np.power(yb_val, 2))
                                        + (np.power(yc_val, 2)
                                           * np.power(ys_val, 2))
                                        + (np.power(yu_val, 2)
                                           * np.power(yd_val, 2))))
                                + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                + np.power(ye_val, 4))) # end trace
                       + (((16 * np.power(g3_val, 2))
                           + ((4 / 5) * np.power(g1_val, 2))) # Tr(Yu^2)
                          * (np.power(yt_val, 2) + np.power(yc_val, 2)
                             + np.power(yu_val, 2))) # end trace
                       + (((16 * np.power(g3_val, 2))
                           - ((2 / 5) * np.power(g1_val, 2))) # Tr(Yd^2)
                          * (np.power(yb_val, 2) + np.power(ys_val, 2)
                             + np.power(yd_val, 2))) # end trace
                       + ((6 / 5) * np.power(g1_val, 2) # Tr(Ye^2)
                          (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))) # end trace
                       + ((15 / 2) * np.power(g2_val, 4))
                       + ((9 / 5) * np.power(g1_val, 2) * np.power(g2_val, 2))
                       + ((207 / 50) * np.power(g1_val, 4))))\
            + (mu_val * (((-12) # Tr(3au*Yu^3 + 3ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + ae*Ye^3)
                          * ((3 * ((at_val * np.power(yt_val, 3))
                                   + (ac_val * np.power(yc_val, 3))
                                   + (au_val * np.power(yu_val, 3))
                                   + (ab_val * np.power(yb_val, 3))
                                   + (as_val * np.power(ys_val, 3))
                                   + (ad_val * np.power(yd_val, 3))))
                             + ((at_val * np.power(yb_val, 2) * yt_val)
                                + (ac_val * np.power(ys_val, 2) * yc_val)
                                + (au_val * np.power(yd_val, 2) * yu_val))
                             + ((ab_val * np.power(yt_val, 2) * yb_val)
                                + (as_val * np.power(yc_val, 2) * ys_val)
                                + (ad_val * np.power(yu_val, 2) * yd_val))
                             + ((atau_val * np.power(ytau_val, 3))
                                + (amu_val * np.power(ymu_val, 3))
                                + (ae_val * np.power(ye_val, 3))))) # end trace
                         + (((32 * np.power(g3_val, 2))
                             + ((8 / 5) * np.power(g1_val, 2))) # Tr(au*Yu)
                            * ((at_val * yt_val) + (ac_val * yc_val)
                               + (au_val * yu_val))) # end trace
                         + (((32 * np.power(g3_val, 2))
                             - ((4 / 5) * np.power(g1_val, 2))) # Tr(ad*Yd)
                            * ((ab_val * yb_val) + (as_val * ys_val)
                               + (ad_val * yd_val))) # end trace
                         + ((12 / 5) * np.power(g1_val, 2) # Tr(ae*Ye)
                            * ((atau_val * ytau_val) + (amu_val * ymu_val)
                               + (ae_val * ye_val))) # end trace
                         - (((32 * np.power(g3_val, 2) * M3_val)
                             + ((8 / 5) * np.power(g1_val, 2) * M1_val)) # Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))
                         - (((32 * np.power(g3_val, 2) * M3_val) # end trace
                             - ((4 / 5) * np.power(g1_val, 2) * M1_val)) # Tr(Yd^2)
                            * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2))) # end trace
                         - ((12 / 5) * np.power(g1_val, 2) * M1_val # Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2))) # end trace
                         - (30 * np.power(g2_val, 4) * M2_val)
                         - ((18 / 5) * np.power(g1_val, 2)
                            * np.power(g2_val, 2)
                            * (M1_val + M2_val))
                         - ((414 / 25) * np.power(g1_val, 4) * M1_val)))

        # Total b beta function
        db_dt = (1 / t) * ((loop_fac * db_dt_1l)
                           + (loop_fac_sq * db_dt_2l))

        ##### Scalar squared masses #####
        # Introduce S, S', and sigma terms
        S_val = mHu_sq_val - mHd_sq_val + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val\
            - mL3_sq_val - mL2_sq_val - mL1_sq_val\
            - (2 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))\
            + mD3_sq_val + mD2_sq_val + mD1_sq_val\
            + mE3_sq_val + mE2_sq_val + mE1_sq_val

        # Tr(-(3mHu^2 + mQ^2) * Yu^2 + 4Yu^2 * mU^2 + (3mHd^2 - mQ^2) * Yd^2
        #    - 2Yd^2 * mD^2 + (mHd^2 + mL^2) * Ye^2 - 2Ye^2 * mE^2)
        Spr_val = (((-1) * ((((3 * mHu_sq_val) + mQ3_sq_val)
                             * np.power(yt_val, 2))
                            + (((3 * mHu_sq_val) + mQ2_sq_val)
                               * np.power(yc_val, 2))
                            + (((3 * mHu_sq_val) + mQ1_sq_val)
                               * np.power(yu_val, 2))))
                   + (4 * np.power(yt_val, 2) * mU3_sq_val)
                   + (4 * np.power(yc_val, 2) * mU2_sq_val)
                   + (4 * np.power(yu_val, 2) * mU1_sq_val)
                   + ((((3 * mHd_sq_val) - mQ3_sq_val) * np.power(yb_val, 2))
                      + (((3 * mHd_sq_val) - mQ2_sq_val)
                         * np.power(ys_val, 2))
                      + (((3 * mHd_sq_val) - mQ1_sq_val)
                         * np.power(yd_val, 2)))
                   - (2 * ((mD3_sq_val * np.power(yb_val, 2))
                           + (mD2_sq_val * np.power(ys_val, 2))
                           + (mD1_sq_val * np.power(yd_val, 2))))
                   + (((mHd_sq_val + mL3_sq_val) * np.power(ytau_val, 2))
                      + ((mHd_sq_val + mL2_sq_val) * np.power(ymu_val, 2))
                      + ((mHd_sq_val + mL1_sq_val) * np.power(ye_val, 2)))
                   - (2 * ((np.power(ytau_val, 2) * mE3_sq_val)
                           + (np.power(ymu_val, 2) * mE2_sq_val)
                           + (np.power(ye_val, 2) * mE1_sq_val))))\ # end trace
            + ((((3 / 2) * np.power(g2_val, 2))
                + ((3 / 10) * np.power(g1_val, 2)))
               * (mHu_sq_val - mHd_sq_val # Tr(mL^2)
                  - (mL3_sq_val + mL2_sq_val + mL1_sq_val))) # end trace
            + ((((8 / 3) * np.power(g3_val, 2))
                + ((3 / 2) * np.power(g2_val, 2))
                + ((1 / 30) * np.power(g1_val, 2))) # Tr (mQ^2)
               * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val)) # end trace
            - ((((16 / 3) * np.power(g3_val, 2))
                + ((16 / 15) * np.power(g1_val, 2))) # Tr (mU^2)
               * (mU3_sq_val + mU2_sq_val + mU1_sq_val)) # end trace
            + ((6 / 5) * np.power(g1_val, 2) # Tr(mE^2)
               * (mE3_sq_val + mE2_sq_val + mE1_sq_val)) # end trace

        sigma1 = (1 / 5) * np.power(g1_val, 2)\
            * ((3 * (mHu_sq_val + mHd_sq_val)) # Tr(mQ^2 + 3mL^2 + 8mU^2 + 2mD^2 + 6mE^2)
               + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
               + (3 * (mL3_sq_val + mL2_sq_val + mL1_sq_val))
               + (8 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
               + (2 * (mD3_sq_val + mD2_sq_val + mD1_sq_val))
               + (6 * (mE3_sq_val + mE2_sq_val + mE1_sq_val))) # end trace

        sigma2 = np.power(g2_val, 2)\
            * (mHu_sq_val + mHd_sq_val # Tr(3mQ^2 + mL^2)
               + (3 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
               + mL3_sq_val + mL2_sq_val + mL1_sq_val) # end trace

        sigma3 = np.power(g3_val, 2)\ # Tr(2mQ^2 + mU^2 + mD^2)
            * ((2 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
               + mU3_sq_val + mU2_sq_val + mU1_sq_val
               + mD3_sq_val + mD2_sq_val + mD1_sq_val) # end trace

        # 1 loop part of Higgs squared masses
        dmHu_sq_dt_1l = 6\ # Tr((mHu^2 + mQ^2) * Yu^2 + Yu^2 * mU^2 + au^2)
            * (((mHu_sq_val + mQ3_sq_val) * np.power(yt_val, 2))
               + ((mHu_sq_val + mQ2_sq_val) * np.power(yc_val, 2))
               + ((mHu_sq_val + mQ1_sq_val) * np.power(yu_val, 2))
               + (mU3_sq_val * np.power(yt_val, 2))
               + (mU2_sq_val * np.power(yc_val, 2))
               + (mU1_sq_val * np.power(yu_val, 2))
               + np.power(at_val, 2) + np.power(ac_val, 2)
               + np.power(au_val, 2))\ # end trace
            - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))\
            - ((6 / 5) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            + ((3 / 5) * np.power(g1_val, 2) * S_val)

        # Tr (6(mHd^2 + mQ^2) * Yd^2 + 6Yd^2*mD^2 + 2(mHd^2 + mL^2) * Ye^2
        #     + 2(Ye^2 * mE^2) + 6ad^2 + 2ae^2)
        dmHd_sq_dt_1l = (6 * (((mHd_sq_val + mQ3_sq_val) * np.power(yb_val, 2))
                              + ((mHd_sq_val + mQ2_sq_val)
                                 * np.power(ys_val, 2))
                              + ((mHd_sq_val + mQ1_sq_val)
                                 * np.power(yd_val, 2)))
                         + (6 * ((mD3_sq_val * np.power(yb_val, 2))
                                 + (mD2_sq_val * np.power(ys_val, 2))
                                 + (mD1_sq_val * np.power(yd_val, 2))))
                         + (2 * (((mHd_sq_val + mL3_sq_val)
                                  * np.power(ytau_val, 2))
                                 + ((mHd_sq_val + mL2_sq_val)
                                    * np.power(ymu_val, 2))
                                 + ((mHd_sq_val + mL1_sq_val)
                                    * np.power(ye_val, 2))))
                         + (2 * ((mE3_sq_val * np.power(ytau_val, 2))
                                 + (mE2_sq_val * np.power(ymu_val, 2))
                                 + (mE1_sq_val * np.power(ye_val, 2))))
                         + (6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                 + np.power(ad_val, 2)))
                         + (2 * (np.power(atau_val, 2) + np.power(amu_val, 2)
                                 + np.power(ae_val, 2))))\ # end trace
            - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
            - ((6 / 5) * np.power(g1_val, 2) * np.power(M1_val, 2))
            - ((3 / 5) * np.power(g1_val, 2) * S_val)

        # 2 loop part of Higgs squared masses
        dmHu_sq_dt_2l = ((-6) # Tr(6(mHu^2 + mQ^2)*Yu^4 + 6Yu^4 * mU^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2 * Yd^2 + Yu^2 * Yd^2 * mU^2 + Yu^2 * Yd^2 * mQ^2 + Yu^2 * Yd^2 * mD^2 + 12au^2 * Yu^2 + ad^2 * Yu^2 + Yd^2 * au^2 + 2ad * Yd * Yu * au)
                         * ((6 * (((mHu_sq_val + mQ3_sq_val)
                                   * np.power(yt_val, 4))
                                  + ((mHu_sq_val + mQ2_sq_val)
                                     * np.power(yc_val, 4))
                                  + ((mHu_sq_val + mQ1_sq_val)
                                     * np.power(yu_val, 4))))
                            + (6 * ((mU3_sq_val * np.power(yt_val, 4))
                                    + (mU2_sq_val * np.power(yc_val, 4))
                                    + (mU1_sq_val * np.power(yu_val, 4))))
                            + ((mHu_sq_val + mHd_sq_val + mQ3_sq_val)
                               * np.power(yt_val, 2) * np.power(yb_val, 2))
                            + ((mHu_sq_val + mHd_sq_val + mQ2_sq_val)
                               * np.power(yc_val, 2) * np.power(ys_val, 2))
                            + ((mHu_sq_val + mHd_sq_val + mQ1_sq_val)
                               * np.power(yu_val, 2) * np.power(yd_val, 2))
                            + ((mU3_sq_val + mQ3_sq_val + mD3_sq_val)
                               * np.power(yt_val, 2) * np.power(yb_val, 2))
                            + ((mU2_sq_val + mQ2_sq_val + mD2_sq_val)
                               * np.power(yc_val, 2) * np.power(ys_val, 2))
                            + ((mU1_sq_val + mQ1_sq_val + mD1_sq_val)
                               * np.power(yu_val, 2) * np.power(yd_val, 2))
                            + (12 * ((np.power(at_val, 2)
                                      * np.power(yt_val, 2))
                                     + (np.power(ac_val, 2)
                                        * np.power(yc_val, 2))
                                     + (np.power(au_val, 2)
                                        * np.power(yu_val, 2))))
                            + (np.power(ab_val, 2) * np.power(yt_val, 2))
                            + (np.power(as_val, 2) * np.power(yc_val, 2))
                            + (np.power(ad_val, 2) * np.power(yu_val, 2))
                            + (np.power(yb_val, 2) * np.power(at_val, 2))
                            + (np.power(ys_val, 2) * np.power(ac_val, 2))
                            + (np.power(yd_val, 2) * np.power(au_val, 2))
                            + (2 * ((yb_val * ab_val * at_val * yt_val)
                                    + (ys_val * as_val * ac_val * yc_val)
                                    + (yd_val * ad_val * au_val * yu_val)))))\  # end trace
            + (((32 * np.power(g3_val, 2)) + ((8 / 5) * np.power(g1_val, 2))) # Tr((mHu^2 + mQ^2 + mU^2) * Yu^2 + au^2)
               * (((mHu_sq_val + mQ3_sq_val + mU3_sq_val)
                   * np.power(yt_val, 2))
                  + ((mHu_sq_val + mQ2_sq_val + mU2_sq_val)
                     * np.power(yc_val, 2))
                  + ((mHu_sq_val + mQ1_sq_val + mU1_sq_val)
                     * np.power(yu_val, 2))
                  + np.power(at_val, 2) + np.power(ac_val, 2)
                  + np.power(au_val, 2)))\ # end trace
            + (32 * np.power(g3_val, 2)
               * ((2 * np.power(M3_val, 2) # Tr(Yu^2)
                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                      + np.power(yu_val, 2))) # end trace
                  - (2 * M3_val # Tr(Yu*au)
                     * ((yt_val * at_val) + (yc_val * ac_val)
                        + (yu_val * au_val)))))\ # end trace
            + ((8 / 5) * np.power(g1_val, 2)
               * ((2 * np.power(M1_val, 2) # Tr(Yu^2)
                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                      + np.power(yu_val, 2))) # end trace
                  - (2 * M1_val # Tr(Yu*au)
                     * ((yt_val * at_val) + (yc_val * ac_val)
                        + (yu_val * au_val)))))\ # end trace
            + ((6 / 5) * np.power(g1_val, 2) * Spr_val)\
            + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))\
            + ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
               * (np.power(M2_val, 2) + np.power(M1_val, 2)
                  + (M1_val * M2_val)))\
            + ((621 / 25) * np.power(g1_val, 4) * np.power(M1_val, 2))\
            + (3 * np.power(g2_val, 2) * sigma2)\
            + ((3 / 5) * np.power(g1_val, 2) * sigma1)

        dmHd_sq_dt_2l = ((-6) # Tr(6(mHd^2 + mQ^2)*Yd^4 + 6Yd^4 * mD^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2 * Yd^2 + Yu^2 * Yd^2 * mU^2 + Yu^2 * Yd^2 * mQ^2 + Yu^2 * Yd^2 * mD^2 + 2(mHd^2 + mL^2) * Ye^4 + 2Ye^4 * mE^2 + 12ad^2 * Yd^2 + ad^2 * Yu^2 + Yd^2 * au^2 + 2ad * Yd * Yu * au + 4ae^2 * Ye^2)
                         * ((6 * (((mHd_sq_val + mQ3_sq_val)
                                   * np.power(yb_val, 4))
                                  + ((mHd_sq_val + mQ2_sq_val)
                                     * np.power(ys_val, 4))
                                  + ((mHd_sq_val + mQ1_sq_val)
                                     * np.power(yd_val, 4))))
                            + (6 * ((mD3_sq_val * np.power(yb_val, 4))
                                    + (mD2_sq_val * np.power(ys_val, 4))
                                    + (mD1_sq_val * np.power(yd_val, 4))))
                            + ((mHu_sq_val + mHd_sq_val + mQ3_sq_val)
                               * np.power(yt_val, 2) * np.power(yb_val, 2))
                            + ((mHu_sq_val + mHd_sq_val + mQ2_sq_val)
                               * np.power(yc_val, 2) * np.power(ys_val, 2))
                            + ((mHu_sq_val + mHd_sq_val + mQ1_sq_val)
                               * np.power(yu_val, 2) * np.power(yd_val, 2))
                            + ((mU3_sq_val + mQ3_sq_val + mD3_sq_val)
                               * np.power(yt_val, 2) * np.power(yb_val, 2))
                            + ((mU2_sq_val + mQ2_sq_val + mD2_sq_val)
                               * np.power(yc_val, 2) * np.power(ys_val, 2))
                            + ((mU1_sq_val + mQ1_sq_val + mD1_sq_val)
                               * np.power(yu_val, 2) * np.power(yd_val, 2))
                            + (2 * (((mHd_sq_val + mL3_sq_val + mE3_sq_val)
                                     * np.power(ytau_val, 4))
                                    + ((mHd_sq_val + mL2_sq_val + mE2_sq_val)
                                       * np.power(ymu_val, 4))
                                    + ((mHd_sq_val + mL1_sq_val + mE1_sq_val)
                                       * np.power(ye_val, 4))))
                            + (12 * ((np.power(ab_val, 2)
                                      * np.power(yb_val, 2))
                                     + (np.power(as_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(ad_val, 2)
                                        * np.power(yd_val, 2))))
                            + (np.power(ab_val, 2) * np.power(yt_val, 2))
                            + (np.power(as_val, 2) * np.power(yc_val, 2))
                            + (np.power(ad_val, 2) * np.power(yu_val, 2))
                            + (np.power(yb_val, 2) * np.power(at_val, 2))
                            + (np.power(ys_val, 2) * np.power(ac_val, 2))
                            + (np.power(yd_val, 2) * np.power(au_val, 2))
                            + (2 * ((yb_val * ab_val * at_val * yt_val)
                                    + (ys_val * as_val * ac_val * yc_val)
                                    + (yd_val * ad_val * au_val * yu_val)
                                    + (2 * ((np.power(atau_val, 2)
                                             * np.power(ytau_val, 2))
                                            + (np.power(amu_val, 2)
                                               * np.power(ymu_val, 2))
                                            + (np.power(ae_val, 2)
                                               * np.power(ye_val, 2))))))))\  # end trace
            + (((32 * np.power(g3_val, 2)) - ((4 / 5) * np.power(g1_val, 2))) # Tr((mHd^2 + mQ^2 + mD^2) * Yd^2 + ad^2)
               * (((mHu_sq_val + mQ3_sq_val + mD3_sq_val)
                   * np.power(yb_val, 2))
                  + ((mHu_sq_val + mQ2_sq_val + mD2_sq_val)
                     * np.power(ys_val, 2))
                  + ((mHu_sq_val + mQ1_sq_val + mD1_sq_val)
                     * np.power(yd_val, 2))
                  + np.power(ab_val, 2) + np.power(as_val, 2)
                  + np.power(ad_val, 2)))\ # end trace
            + (32 * np.power(g3_val, 2)
               * ((2 * np.power(M3_val, 2) # Tr(Yd^2)
                   * (np.power(yb_val, 2) + np.power(ys_val, 2)
                      + np.power(yd_val, 2))) # end trace
                  - (2 * M3_val # Tr(Yd*ad)
                     * ((yb_val * ab_val) + (ys_val * as_val)
                        + (yd_val * ad_val)))))\ # end trace
            - ((4 / 5) * np.power(g1_val, 2)
               * ((2 * np.power(M1_val, 2) # Tr(Yd^2)
                   * (np.power(yb_val, 2) + np.power(ys_val, 2)
                      + np.power(yd_val, 2))) # end trace
                  - (2 * M1_val # Tr(Yd*ad)
                     * ((yb_val * ab_val) + (ys_val * as_val)
                        + (yd_val * ad_val)))))\ # end trace
            + ((12 / 5) * np.power(g1_val, 2)
               * (# Tr((mHd^2 + mL^2 + mE^2) * Ye^2 + ae^2)
                  ((mHd_sq_val + mL3_sq_val + mE3_sq_val)
                   * np.power(ytau_val, 2))
                  + ((mHd_sq_val + mL2_sq_val + mE2_sq_val)
                     * np.power(ymu_val, 2))
                  + ((mHd_sq_val + mL1_sq_val + mE1_sq_val)
                     * np.power(ye_val, 2))
                  + np.power(atau_val, 2) + np.power(amu_val, 2)
                  + np.power(ae_val, 2)))\ # end trace
            - ((6 / 5) * np.power(g1_val, 2) * Spr_val)\
            + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))\
            + ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
               * (np.power(M2_val, 2) + np.power(M1_val, 2)
                  + (M1_val * M2_val)))\
            + ((621 / 25) * np.power(g1_val, 4) * np.power(M1_val, 2))\
            + (3 * np.power(g2_val, 2) * sigma2)\
            + ((3 / 5) * np.power(g1_val, 2) * sigma1)

        # Total Higgs squared mass beta functions
        dmHu_sq_dt = (1 / t) * ((loop_fac * dmHu_sq_dt_1l)
                                + (loop_fac_sq * dmHu_sq_dt_2l))

        dmHd_sq_dt = (1 / t) * ((loop_fac * dmHd_sq_dt_1l)
                                + (loop_fac_sq * dmHd_sq_dt_2l))

        # 1 loop parts of scalar squared masses
        # Left squarks
        dmQ3_sq_dt_1l = ((mQ3_sq_val + (2 * mHu_sq_val))
                         * np.power(yt_val, 2))\
            + ((mQ3_sq_val + (2 * mHd_sq_val)) * np.power(yb_val, 2))\
            + ((np.power(yt_val, 2) + np.power(yb_val, 2)) * mQ3_sq_val)\
            + (2 * np.power(yt_val, 2) * mU3_sq_val)\
            + (2 * np.power(yb_val, 2) * mD3_sq_val)\
            + (2 * np.power(at_val, 2)) + (2 * np.power(ab_val, 2))\
            - ((32 / 3) * np.power(g3_val, 2) * np.power(M3_val, 2))\
            - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))\
            - ((2 / 15) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            + ((1 / 5) * np.power(g1_val, 2) * S_val)

        dmQ2_sq_dt_1l = ((mQ2_sq_val + (2 * mHu_sq_val))
                         * np.power(yc_val, 2))\
            + ((mQ2_sq_val + (2 * mHd_sq_val)) * np.power(ys_val, 2))\
            + ((np.power(yc_val, 2) + np.power(ys_val, 2)) * mQ2_sq_val)\
            + (2 * np.power(yc_val, 2) * mU2_sq_val)\
            + (2 * np.power(ys_val, 2) * mD2_sq_val)\
            + (2 * np.power(ac_val, 2)) + (2 * np.power(as_val, 2))\
            - ((32 / 3) * np.power(g3_val, 2) * np.power(M3_val, 2))\
            - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))\
            - ((2 / 15) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            + ((1 / 5) * np.power(g1_val, 2) * S_val)

        dmQ1_sq_dt_1l = ((mQ1_sq_val + (2 * mHu_sq_val))
                         * np.power(yu_val, 2))\
            + ((mQ1_sq_val + (2 * mHd_sq_val)) * np.power(yd_val, 2))\
            + ((np.power(yu_val, 2) + np.power(yd_val, 2)) * mQ1_sq_val)\
            + (2 * np.power(yu_val, 2) * mU1_sq_val)\
            + (2 * np.power(yd_val, 2) * mD1_sq_val)\
            + (2 * np.power(au_val, 2)) + (2 * np.power(ad_val, 2))\
            - ((32 / 3) * np.power(g3_val, 2) * np.power(M3_val, 2))\
            - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))\
            - ((2 / 15) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            + ((1 / 5) * np.power(g1_val, 2) * S_val)

        # Left leptons
        dmL3_sq_dt_1l = ((mL3_sq_val + (2 * mHd_sq_val))
                         * np.power(ytau_val, 2))\
            + (2 * np.power(ytau_val, 2) * mE3_sq_val)\
            + (np.power(ytau_val, 2) * mL3_sq_val)\
            + (2 * np.power(atau_val, 2))\
            - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))\
            - ((6 / 5) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            - ((3 / 5) * np.power(g1_val, 2) * S_val)

        dmL2_sq_dt_1l = ((mL2_sq_val + (2 * mHd_sq_val))
                         * np.power(ymu_val, 2))\
            + (2 * np.power(ymu_val, 2) * mE2_sq_val)\
            + (np.power(ymu_val, 2) * mL2_sq_val)\
            + (2 * np.power(amu_val, 2))\
            - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))\
            - ((6 / 5) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            - ((3 / 5) * np.power(g1_val, 2) * S_val)

        dmL1_sq_dt_1l = ((mL1_sq_val + (2 * mHd_sq_val))
                         * np.power(ye_val, 2))\
            + (2 * np.power(ye_val, 2) * mE1_sq_val)\
            + (np.power(ye_val, 2) * mL1_sq_val)\
            + (2 * np.power(ae_val, 2))\
            - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))\
            - ((6 / 5) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            - ((3 / 5) * np.power(g1_val, 2) * S_val)

        # Right up-type squarks
        dmU3_sq_dt_1l = (2 * (mU3_sq_val + (2 * mHd_sq_val))
                         * np.power(yt_val, 2))\
            + (4 * np.power(yt_val, 2) * mQ3_sq_val)\
            + (2 * np.power(yt_val, 2) * mU3_sq_val)\
            + (4 * np.power(at_val, 2))\
            - ((32 / 3) * np.power(g3_val, 2) * np.power(M3_val, 2))
            - ((32 / 15) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            - ((4 / 5) * np.power(g1_val, 2) * S_val)

        dmU2_sq_dt_1l = (2 * (mU2_sq_val + (2 * mHd_sq_val))
                         * np.power(yc_val, 2))\
            + (4 * np.power(yc_val, 2) * mQ2_sq_val)\
            + (2 * np.power(yc_val, 2) * mU2_sq_val)\
            + (4 * np.power(ac_val, 2))\
            - ((32 / 3) * np.power(g3_val, 2) * np.power(M3_val, 2))
            - ((32 / 15) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            - ((4 / 5) * np.power(g1_val, 2) * S_val)

        dmU1_sq_dt_1l = (2 * (mU1_sq_val + (2 * mHd_sq_val))
                         * np.power(yu_val, 2))\
            + (4 * np.power(yu_val, 2) * mQ1_sq_val)\
            + (2 * np.power(yu_val, 2) * mU1_sq_val)\
            + (4 * np.power(au_val, 2))\
            - ((32 / 3) * np.power(g3_val, 2) * np.power(M3_val, 2))
            - ((32 / 15) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            - ((4 / 5) * np.power(g1_val, 2) * S_val)

        # Right down-type squarks
        dmD3_sq_dt_1l = (2 * (mD3_sq_val + (2 * mHd_sq_val))
                         * np.power(yb_val, 2))\
            + (4 * np.power(yb_val, 2) * mQ3_sq_val)\
            + (2 * np.power(yb_val, 2) * mU3_sq_val)\
            + (4 * np.power(ab_val, 2))\
            - ((32 / 3) * np.power(g3_val, 2) * np.power(M3_val, 2))
            - ((8 / 15) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            + ((2 / 5) * np.power(g1_val, 2) * S_val)

        dmD2_sq_dt_1l = (2 * (mD2_sq_val + (2 * mHd_sq_val))
                         * np.power(ys_val, 2))\
            + (4 * np.power(ys_val, 2) * mQ2_sq_val)\
            + (2 * np.power(ys_val, 2) * mU2_sq_val)\
            + (4 * np.power(as_val, 2))\
            - ((32 / 3) * np.power(g3_val, 2) * np.power(M3_val, 2))
            - ((8 / 15) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            + ((2 / 5) * np.power(g1_val, 2) * S_val)

        dmD1_sq_dt_1l = (2 * (mD1_sq_val + (2 * mHd_sq_val))
                         * np.power(yd_val, 2))\
            + (4 * np.power(yd_val, 2) * mQ1_sq_val)\
            + (2 * np.power(yd_val, 2) * mU1_sq_val)\
            + (4 * np.power(ad_val, 2))\
            - ((32 / 3) * np.power(g3_val, 2) * np.power(M3_val, 2))
            - ((8 / 15) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            + ((2 / 5) * np.power(g1_val, 2) * S_val)

        # Right leptons
        dmE3_sq_dt_1l = (2 * (mE3_sq_val + (2 * mHd_sq_val))
                         * np.power(ytau_val, 2))\
            + (4 * np.power(ytau_val, 2) * mL3_sq_val)\
            + (2 * np.power(ytau_val, 2) * mE3_sq_val)\
            + (4 * np.power(atau_val, 2))\
            - ((24 / 5) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            + ((6 / 5) * np.power(g1_val, 2) * S_val)

        dmE2_sq_dt_1l = (2 * (mE2_sq_val + (2 * mHd_sq_val))
                         * np.power(ymu_val, 2))\
            + (4 * np.power(ymu_val, 2) * mL2_sq_val)\
            + (2 * np.power(ymu_val, 2) * mE2_sq_val)\
            + (4 * np.power(amu_val, 2))\
            - ((24 / 5) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            + ((6 / 5) * np.power(g1_val, 2) * S_val)

        dmE1_sq_dt_1l = (2 * (mE1_sq_val + (2 * mHd_sq_val))
                         * np.power(ye_val, 2))\
            + (4 * np.power(ye_val, 2) * mL1_sq_val)\
            + (2 * np.power(ye_val, 2) * mE1_sq_val)\
            + (4 * np.power(ae_val, 2))\
            - ((24 / 5) * np.power(g1_val, 2) * np.power(M1_val, 2))\
            + ((6 / 5) * np.power(g1_val, 2) * S_val)

        # 2 loop parts of scalar squared masses
        # Left squarks
        dmQ3_sq_dt_2l = ((-8) * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                         * np.power(yt_val, 4))\
            - (8 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
               * np.power(yt_val, 4))\
            - (((2 * mQ3_sq_val) + (2 * mU3_sq_val) + (4 * mHu_sq_val)) # Tr(3Yu^2)
               * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                      + np.power(yu_val, 2)))\ # end trace
            - (((2 * mQ3_sq_val) + (2 * mD3_sq_val) + (4 * mHd_sq_val)) # Tr(3Yd^2 + Ye^2)
               * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                  + np.power(ye_val, 2)))\ # end trace
            - (6 * np.power(yt_val, 2) # Tr((mQ^2 + mU^2) * Yu^2)
               * (((mQ3_sq_val + mU3_sq_val) * np.power(yt_val, 2))
                  + ((mQ2_sq_val + mU2_sq_val) * np.power(yc_val, 2))
                  + ((mQ1_sq_val + mU1_sq_val) * np.power(yu_val, 2))))\ # end trace
            - (np.power(yb_val, 2) # Tr(6(mQ^2 + mD^2) * Yd^2 + 2(mL^2 + mE^2) * Ye^2)
               * ((6 * (((mQ3_sq_val + mD3_sq_val) * np.power(yb_val, 2))
                        + ((mQ2_sq_val + mD2_sq_val) * np.power(ys_val, 2))
                        + ((mQ1_sq_val + mD1_sq_val) * np.power(yd_val, 2))))
                  + (2 * (((mL3_sq_val + mE3_sq_val) * np.power(ytau_val, 2))
                          + ((mL2_sq_val + mE2_sq_val) * np.power(ymu_val, 2))
                          + ((mL1_sq_val + mE1_sq_val) * np.power(ye_val, 2)))) # end trace
                  ))\
            - (16 * np.power(yt_val, 2) * np.power(at_val, 2))\
            - (16 * np.power(yb_val, 2) * np.power(ab_val, 2))\
            - (np.power(at_val, 2) # Tr(6Yu^2)
               * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                      + np.power(yu_val, 2)))\ # end trace
            - (np.power(yt_val, 2) # Tr(6au^2)
               * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                      + np.power(au_val, 2)))\ # end trace
            - (at_val * yt_val # Tr(12Yu*au)
               * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                       + (yu_val * au_val)))\ # end trace
            - (np.power(ab_val, 2) # Tr(6Yd^2 + 2Ye^2)
               * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  + (2 * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                          + np.power(ye_val, 2)))))\ # end trace
            - (np.power(yb_val, 2) # Tr(6ad^2 + 2ae^2)
               * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                        + np.power(ad_val, 2)))
                  + (2 * (np.power(atau_val, 2) + np.power(amu_val, 2)
                          + np.power(ae_val, 2)))))\ # end trace
            - (2 * ab_val * yb_val # Tr(6Yd * ad + 2Ye * ae)
               * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                        + (yd_val * ad_val)))
                  + (2 * ((ytau_val * atau_val) + (ymu_val * amu_val)
                          + (ye_val * ae_val)))))\ # end trace
            + ((2 / 5) * np.power(g1_val, 2)
               * ((4 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                   * np.power(yt_val, 2))
                  + (4 * np.power(at_val, 2)) - (8 * M1_val * at_val * yt_val)
                  + (8 * np.power(M1_val, 2) * np.power(yt_val, 2))
                  + (2 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                     * np.power(yb_val, 2))
                  + (2 * np.power(ab_val, 2)) - (4 * M1_val * ab_val * yb_val)
                  + (4 * np.power(M1_val, 2) * np.power(yb_val, 2))))\
            + ((2 / 5) * np.power(g1_val, 2) * Spr_val)\
            - ((128 / 3) * np.power(g3_val, 4) * np.power(M3_val, 2))\
            + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
               * (np.power(M3_val, 2) + np.power(M2_val, 2)
                  + (M2_val * M3_val)))\
            + ((32 / 45) * np.power(g3_val, 2) * np.power(g1_val, 2)
               * (np.power(M3_val, 2) + np.power(M1_val, 2)
                  + (M3_val * M1_val)))\
            + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))\
            + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
               * (np.power(M1_val, 2) + np.power(M2_val, 2)
                  + (M1_val * M2_val)))\
            + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))\
            + ((16 / 3) * np.power(g3_val, 2) * sigma3)\
            + (3 * np.power(g2_val, 2) * sigma2)\
            + ((1 / 15) * np.power(g1_val, 2) * sigma1)

        dmQ2_sq_dt_2l =

        dmQ1_sq_dt_2l =

        # Left leptons
        dmL3_sq_dt_2l =

        dmL2_sq_dt_2l =

        dmL1_sq_dt_2l =

        # Right up-type squarks
        dmU3_sq_dt_2l =

        dmU2_sq_dt_2l =

        dmU1_sq_dt_2l =

        # Right down-type squarks
        dmD3_sq_dt_2l =

        dmD2_sq_dt_2l =

        dmD1_sq_dt_2l =

        # Right leptons
        dmE3_sq_dt_2l =

        dmE2_sq_dt_2l =

        dmE1_sq_dt_2l =

        # Total scalar squared mass beta functions
        dmQ3_sq_dt = (1 / t) * ((loop_fac * dmQ3_sq_dt_1l)
                                + (loop_fac_sq * dmQ3_sq_dt_2l))

        dmQ2_sq_dt = (1 / t) * ((loop_fac * dmQ2_sq_dt_1l)
                                + (loop_fac_sq * dmQ2_sq_dt_2l))

        dmQ1_sq_dt = (1 / t) * ((loop_fac * dmQ1_sq_dt_1l)
                                + (loop_fac_sq * dmQ1_sq_dt_2l))

        dmL3_sq_dt = (1 / t) * ((loop_fac * dmL3_sq_dt_1l)
                                + (loop_fac_sq * dmL3_sq_dt_2l))

        dmL2_sq_dt = (1 / t) * ((loop_fac * dmL2_sq_dt_1l)
                                + (loop_fac_sq * dmL2_sq_dt_2l))

        dmL1_sq_dt = (1 / t) * ((loop_fac * dmL1_sq_dt_1l)
                                + (loop_fac_sq * dmL1_sq_dt_2l))

        dmU3_sq_dt = (1 / t) * ((loop_fac * dmU3_sq_dt_1l)
                                + (loop_fac_sq * dmU3_sq_dt_2l))

        dmU2_sq_dt = (1 / t) * ((loop_fac * dmU2_sq_dt_1l)
                                + (loop_fac_sq * dmU2_sq_dt_2l))

        dmU1_sq_dt = (1 / t) * ((loop_fac * dmU1_sq_dt_1l)
                                + (loop_fac_sq * dmU1_sq_dt_2l))

        dmD3_sq_dt = (1 / t) * ((loop_fac * dmD3_sq_dt_1l)
                                + (loop_fac_sq * dmD3_sq_dt_2l))

        dmD2_sq_dt = (1 / t) * ((loop_fac * dmD2_sq_dt_1l)
                                + (loop_fac_sq * dmD2_sq_dt_2l))

        dmD1_sq_dt = (1 / t) * ((loop_fac * dmD1_sq_dt_1l)
                                + (loop_fac_sq * dmD1_sq_dt_2l))

        dmE3_sq_dt = (1 / t) * ((loop_fac * dmE3_sq_dt_1l)
                                + (loop_fac_sq * dmE3_sq_dt_2l))

        dmE2_sq_dt = (1 / t) * ((loop_fac * dmE2_sq_dt_1l)
                                + (loop_fac_sq * dmE2_sq_dt_2l))

        dmE1_sq_dt = (1 / t) * ((loop_fac * dmE1_sq_dt_1l)
                                + (loop_fac_sq * dmE1_sq_dt_2l))

        ##### Tanb RGE from arXiv:hep-ph/0112251 in R_xi=1 Feynman gauge #####
        dtanb_dt = (tanbval / t) * (((3 / (16 * (np.pi ** 2))) * (((ybval ** 2) - (ytval **2)))) - ((3 / (((16 * (np.pi ** 2))) ** 2)) * ((3 / 40) * (g1val ** 2) + (9 / 8) * (g2val ** 2))) - ((1 / (((16 * (np.pi ** 2))) ** 2)) * ((-9) * ((ytval ** 4) - (ybval ** 4)) + 6 * ((ytval ** 2) * ((8 * (g3val ** 2) / 3) + (2 * (g1val ** 2) / 15)) - (ybval ** 2) * ((8 * (g3val ** 2) / 3) - ((g1val ** 2) / 15))))))


        # Collect all for return
        dxdt = [dg1_dt, dg2_dt, dg3_dt, dM1_dt, dM2_dt, dM3_dt, dmu_dt, dyt_dt,
                dyc_dt, dyu_dt, dyb_dt, dys_dt, dyd_dt, dytau_dt, dymu_dt,
                dye_dt, dat_dt, dac_dt, dau_dt, dab_dt, das_dt, dad_dt,
                datau_dt, damu_dt, dae_dt, db_dt, dmHu_sq_dt, dmHd_sq_dt,
                dmQ1_sq_dt, dmQ2_sq_dt, dmQ3_sq_dt, dmL1_sq_dt, dmL2_sq_dt,
                dmL3_sq_dt, dmU1_sq_dt, dmU2_sq_dt, dmU3_sq_dt, dmD1_sq_dt,
                dmD2_sq_dt, dmD3_sq_dt, dmE1_sq_dt, dmE2_sq_dt, dmE3_sq_dt,
                dtanb_dt]
        return dxdt

    # Set up domains for solve_ivp
    t_span = np.array([float(QGUT), float(low_Q_val)])
    test_vals = np.logspace(np.log10(QGUT),
                            np.log10(low_Q_val), numpoints)

    # Now solve
    sol = solve_ivp(soft_odes, t_span, GUT_BCs, t_eval = test_vals,
                    dense_output=True, method='DOP853', atol=1e-9, rtol=1e-9)

    t = sol.t
    x = sol.y
    return x
