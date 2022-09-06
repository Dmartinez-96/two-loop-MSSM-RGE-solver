#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:49:31 2022

Two-Loop Renormalization Group Equation solver for the MSSM for a user-supplied
    set of input parameters.

@author: Dakotah Martinez
"""

from scipy.integrate import solve_ivp
from scipy.special import spence
import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
import pyslha
import tempfile

##### RGEs #####

def my_RGE_solver(GUT_BCs, my_QGUT, low_Q_val=1000.0):
    """
    Use scipy.integrate to evolve MSSM RGEs and collect solution vectors.

    Parameters
    ----------
    low_BCs : Array of floats.
        GUT scale boundary conditions for RGEs.
    my_QGUT : Float.
        Highest value for t parameter to run to in solution,
            typically unification scale from SoftSUSY.
    low_Q_val : Float.
        Lowest value for t parameter to run to in solution. Default is 1 TeV.

    Returns
    -------
    Array of floats.
        Return solutions to system of RGEs.

    """
    def my_odes(t, x):
        """
        Define two-loop RGEs for soft terms.

        Parameters
        ----------
        x : Array of floats.
            Numerical solutions to RGEs. The order of entries in x is:
              (0: g1, 1: g2, 2: g3, 3: M1, 4: M2, 5: M3, 6: mu, 7: yt, 8: yc,
               9: yu, 10: yb, 11: ys, 12: yd, 13: ytau, 14: ymu, 15: ye,
               16: at, 17: ac, 18: au, 19: ab, 20: as, 21: ad, 22: atau,
               23: amu, 24: ae, 25: mHu^2, 26: mHd^2, 27: mQ1^2,
               28: mQ2^2, 29: mQ3^2, 30: mL1^2, 31: mL2^2, 32: mL3^2,
               33: mU1^2, 34: mU2^2, 35: mU3^2, 36: mD1^2, 37: mD2^2,
               38: mD3^2, 39: mE1^2, 40: mE2^2, 41: mE3^2)
        t : Array of evaluation renormalization scales.
            t = Q values for numerical solutions.

        Returns
        -------
        Array of floats.
            Return all soft RGEs evaluated at current t value.

        """
        # Unification scale is acquired from running a BM point through
        # SoftSUSY, then GUT scale boundary conditions are acquired from
        # SoftSUSY so that all three generations of Yukawas (assumed
        # to be diagonalized) are accounted for. A universal boundary condition
        # is used for soft scalar trilinear couplings a_i=y_i*A_i.
        # The soft b^(ij) mass^2 term is defined as b=B*mu, but is computed
        # in a later iteration.
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
        mHu_sq_val = x[25]
        mHd_sq_val = x[26]
        mQ1_sq_val = x[27]
        mQ2_sq_val = x[28]
        mQ3_sq_val = x[29]
        mL1_sq_val = x[30]
        mL2_sq_val = x[31]
        mL3_sq_val = x[32]
        mU1_sq_val = x[33]
        mU2_sq_val = x[34]
        mU3_sq_val = x[35]
        mD1_sq_val = x[36]
        mD2_sq_val = x[37]
        mD3_sq_val = x[38]
        mE1_sq_val = x[39]
        mE2_sq_val = x[40]
        mE3_sq_val = x[41]

        ##### Gauge couplings and gaugino masses #####
        # 1 loop parts
        dg1_dt_1l = b_1l[0] * np.power(g1_val, 3)

        dg2_dt_1l = b_1l[1] * np.power(g2_val, 3)

        dg3_dt_1l = b_1l[2] * np.power(g3_val, 3)

        dM1_dt_1l = b_1l[0] * np.power(g1_val, 2) * M1_val

        dM2_dt_1l = b_1l[1] * np.power(g2_val, 2) * M2_val

        dM3_dt_1l = b_1l[2] * np.power(g3_val, 2) * M3_val

        # 2 loop parts
        dg1_dt_2l = (np.power(g1_val, 3)
                     * ((b_2l[0][0] * np.power(g1_val, 2))
                        + (b_2l[0][1] * np.power(g2_val, 2))
                        + (b_2l[0][2] * np.power(g3_val, 2))# Tr(Yu^2)
                        - (c_2l[0][0] * (np.power(yt_val, 2)
                                         + np.power(yc_val, 2)
                                         + np.power(yu_val, 2)))# end trace, begin Tr(Yd^2)
                        - (c_2l[0][1] * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))# end trace, begin Tr(Ye^2)
                        - (c_2l[0][2] * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace

        dg2_dt_2l = (np.power(g2_val, 3)
                     * ((b_2l[1][0] * np.power(g1_val, 2))
                        + (b_2l[1][1] * np.power(g2_val, 2))
                        + (b_2l[1][2] * np.power(g3_val, 2))# Tr(Yu^2)
                        - (c_2l[1][0] * (np.power(yt_val, 2)
                                         + np.power(yc_val, 2)
                                         + np.power(yu_val, 2)))# end trace, begin Tr(Yd^2)
                        - (c_2l[1][1] * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))# end trace, begin Tr(Ye^2)
                        - (c_2l[1][2] * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace

        dg3_dt_2l = (np.power(g3_val, 3)
                     * ((b_2l[2][0] * np.power(g1_val, 2))
                        + (b_2l[2][1] * np.power(g2_val, 2))
                        + (b_2l[2][2] * np.power(g3_val, 2))# Tr(Yu^2)
                        - (c_2l[2][0] * (np.power(yt_val, 2)
                                         + np.power(yc_val, 2)
                                         + np.power(yu_val, 2)))# end trace, begin Tr(Yd^2)
                        - (c_2l[2][1] * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))# end trace, begin Tr(Ye^2)
                        - (c_2l[2][2] * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace

        dM1_dt_2l = (2 * np.power(g1_val, 2)
                     * (((b_2l[0][0] * np.power(g1_val, 2) * (M1_val + M1_val))
                         + (b_2l[0][1] * np.power(g2_val, 2)
                            * (M1_val + M2_val))
                         + (b_2l[0][2] * np.power(g3_val, 2)
                            * (M1_val + M3_val)))# Tr(Yu*au)
                        + ((c_2l[0][0] * (((yt_val * at_val)
                                           + (yc_val * ac_val)
                                           + (yu_val * au_val))# end trace, begin Tr(Yu^2)
                                          - (M1_val * (np.power(yt_val, 2)
                                                       + np.power(yc_val, 2)
                                                       + np.power(yu_val, 2)))# end trace
                                          )))# Tr(Yd*ad)
                        + ((c_2l[0][1] * (((yb_val * ab_val)
                                           + (ys_val * as_val)
                                           + (yd_val * ad_val))# end trace, begin Tr(Yd^2)
                                          - (M1_val * (np.power(yb_val, 2)
                                                       + np.power(ys_val, 2)
                                                       + np.power(yd_val, 2)))# end trace
                                          )))# Tr(Ye*ae)
                        + ((c_2l[0][2] * (((ytau_val * atau_val)
                                           + (ymu_val * amu_val)
                                           + (ye_val * ae_val))# end trace, begin Tr(Ye^2)
                                          - (M1_val * (np.power(ytau_val, 2)
                                                       + np.power(ymu_val, 2)
                                                       + np.power(ye_val, 2)))
                                          )))))

        dM2_dt_2l = (2 * np.power(g2_val, 2)
                     * (((b_2l[1][0] * np.power(g1_val, 2) * (M2_val + M1_val))
                         + (b_2l[1][1] * np.power(g2_val, 2)
                            * (M2_val + M2_val))
                         + (b_2l[1][2] * np.power(g3_val, 2)
                            * (M2_val + M3_val)))# Tr(Yu*au)
                        + ((c_2l[1][0] * (((yt_val * at_val)
                                           + (yc_val * ac_val)
                                           + (yu_val * au_val))# end trace, begin Tr(Yu^2)
                                          - (M2_val * (np.power(yt_val, 2)
                                                       + np.power(yc_val, 2)
                                                       + np.power(yu_val, 2)))# end trace
                                          )))# Tr(Yd*ad)
                        + ((c_2l[1][1] * (((yb_val * ab_val)
                                           + (ys_val * as_val)
                                           + (yd_val * ad_val))# end trace, begin Tr(Yd^2)
                                          - (M2_val * (np.power(yb_val, 2)
                                                       + np.power(ys_val, 2)
                                                       + np.power(yd_val, 2)))# end trace
                                          )))# Tr(Ye*ae)
                        + ((c_2l[1][2] * (((ytau_val * atau_val)
                                           + (ymu_val * amu_val)
                                           + (ye_val * ae_val))# end trace, begin Tr(Ye^2)
                                          - (M2_val * (np.power(ytau_val, 2)
                                                       + np.power(ymu_val, 2)
                                                       + np.power(ye_val, 2)))# end trace
                                          )))))

        dM3_dt_2l = (2 * np.power(g3_val, 2)
                     * (((b_2l[2][0] * np.power(g1_val, 2) * (M3_val + M1_val))
                         + (b_2l[2][1] * np.power(g2_val, 2)
                            * (M3_val + M2_val))
                         + (b_2l[2][2] * np.power(g3_val, 2)
                            * (M3_val + M3_val)))# Tr(Yu*au)
                        + ((c_2l[2][0] * (((yt_val * at_val)
                                           + (yc_val * ac_val)
                                           + (yu_val * au_val))# end trace, begin Tr(Yu^2)
                                          - (M3_val * (np.power(yt_val, 2)
                                                       + np.power(yc_val, 2)
                                                       + np.power(yu_val, 2)))# end trace
                                          )))# Tr(Yd*ad)
                        + ((c_2l[2][1] * (((yb_val * ab_val)
                                           + (ys_val * as_val)
                                           + (yd_val * ad_val))# end trace, begin Tr(Yd^2)
                                          - (M3_val * (np.power(yb_val, 2)
                                                       + np.power(ys_val, 2)
                                                       + np.power(yd_val, 2)))# end trace
                                          )))# Tr(Ye*ae)
                        + ((c_2l[2][2] * (((ytau_val * atau_val)
                                           + (ymu_val * amu_val)
                                           + (ye_val * ae_val))# end trace, begin Tr(Ye^2)
                                          - (M3_val * (np.power(ytau_val, 2)
                                                       + np.power(ymu_val, 2)
                                                       + np.power(ye_val, 2)))# end trace
                                          )))))

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

        dmu_dt = 0.0 * mu_val
        ##### Higgsino mass parameter mu #####
        # 1 loop part
        #dmu_dt_1l = (mu_val# Tr(3Yu^2 + 3Yd^2 + Ye^2)
        #             * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
        #                      + np.power(yu_val, 2) + np.power(yb_val, 2)
        #                      + np.power(ys_val, 2) + np.power(yd_val, 2)))
        #                + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
        #                   + np.power(ye_val, 2))# end trace
        #                - (3 * np.power(g2_val, 2))
        #                - ((3 / 5) * np.power(g1_val, 2))))

        # 2 loop part
        #dmu_dt_2l = (mu_val# Tr(3Yu^4 + 3Yd^4 + (2Yu^2*Yd^2) + Ye^4)
        #             * ((-3 * ((3 * (np.power(yt_val, 4) + np.power(yc_val, 4)
        #                             + np.power(yu_val, 4)
        #                             + np.power(yb_val, 4)
        #                             + np.power(ys_val, 4)
        #                             + np.power(yd_val, 4)))
        #                       + (2 * ((np.power(yt_val, 2)
        #                                * np.power(yb_val, 2))
        #                               + (np.power(yc_val, 2)
        #                                  * np.power(ys_val, 2))
        #                               + (np.power(yu_val, 2)
        #                                  * np.power(yd_val, 2))))
        #                       + (np.power(ytau_val, 4) + np.power(ymu_val, 4)
        #                          + np.power(ye_val, 4))))# end trace
        #                + (((16 * np.power(g3_val, 2))
        #                    + (4 * np.power(g1_val, 2) / 5))# Tr(Yu^2)
        #                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
        #                      + np.power(yu_val, 2)))# end trace
        #                + (((16 * np.power(g3_val, 2))
        #                    - (2 * np.power(g1_val, 2) / 5))# Tr(Yd^2)
        #                   * (np.power(yb_val, 2) + np.power(ys_val, 2)
        #                      + np.power(yd_val, 2)))# end trace
        #                + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
        #                   * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
        #                      + np.power(ye_val, 2)))# end trace
        #                + ((15 / 2) * np.power(g2_val, 4))
        #                + ((9 / 5) * np.power(g1_val, 2)
        #                   * np.power(g2_val, 2))
        #                + ((207 / 50) * np.power(g1_val, 4))))

        # Total mu beta function
        #dmu_dt = (1 / t) * ((loop_fac * dmu_dt_1l)
        #                    + (loop_fac_sq * dmu_dt_2l))

        ##### Yukawa couplings for all 3 generations, assumed diagonalized#####
        # 1 loop parts
        dyt_dt_1l = (yt_val# Tr(3Yu^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (3 * (np.power(yt_val, 2)))
                        + np.power(yb_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))

        dyc_dt_1l = (yc_val# Tr(3Yu^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (3 * (np.power(yc_val, 2)))
                        + np.power(ys_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))

        dyu_dt_1l = (yu_val# Tr(3Yu^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (3 * (np.power(yu_val, 2)))
                        + np.power(yd_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))

        dyb_dt_1l = (yb_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(yb_val, 2))) + np.power(yt_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))

        dys_dt_1l = (ys_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(ys_val, 2))) + np.power(yc_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))

        dyd_dt_1l = (yd_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(yd_val, 2))) + np.power(yu_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))

        dytau_dt_1l = (ytau_val# Tr(3Yd^2 + Ye^2)
                       * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2))# end trace
                          + (3 * (np.power(ytau_val, 2)))
                          - (3 * np.power(g2_val, 2))
                          - ((9 / 5) * np.power(g1_val, 2))))

        dymu_dt_1l = (ymu_val# Tr(3Yd^2 + Ye^2)
                      * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))
                         + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                            + np.power(ye_val, 2))# end trace
                         + (3 * (np.power(ymu_val, 2)))
                         - (3 * np.power(g2_val, 2))
                         - ((9 / 5) * np.power(g1_val, 2))))

        dye_dt_1l = (ye_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(ye_val, 2)))
                        - (3 * np.power(g2_val, 2))
                        - ((9 / 5) * np.power(g1_val, 2))))

        # 2 loop parts
        dyt_dt_2l = (yt_val # Tr(3Yu^4 + (Yu^2*Yd^2))
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2)))
                         )# end trace
                        - (np.power(yb_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))# end trace
                        - (9 * np.power(yt_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (4 * np.power(yt_val, 4))
                        - (2 * np.power(yb_val, 4))
                        - (2 * np.power(yb_val, 2) * np.power(yt_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((6 * np.power(g2_val, 2))
                            + ((2 / 5) * np.power(g1_val, 2)))
                           * np.power(yt_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2) * np.power(yb_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))

        dyc_dt_2l = (yc_val # Tr(3Yu^4 + (Yu^2*Yd^2))
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2)))
                         )#end trace
                        - (np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))# end trace
                        - (9 * np.power(yc_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (4 * np.power(yc_val, 4))
                        - (2 * np.power(ys_val, 4))
                        - (2 * np.power(ys_val, 2)
                           * np.power(yc_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((6 * np.power(g2_val, 2))
                            + ((2 / 5) * np.power(g1_val, 2)))
                           * np.power(yc_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2) * np.power(ys_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))

        dyu_dt_2l = (yu_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2)))
                         )# end trace
                        - (np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))
                        - (9 * np.power(yu_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        - (4 * np.power(yu_val, 4))
                        - (2 * np.power(yd_val, 4))
                        - (2 * np.power(yd_val, 2) * np.power(yu_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((6 * np.power(g2_val, 2))
                            + ((2 / 5) * np.power(g1_val, 2)))
                           * np.power(yu_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2) * np.power(yd_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))

        dyb_dt_2l = (yb_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(yt_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (3 * np.power(yb_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(yb_val, 4))
                        - (2 * np.power(yt_val, 4))
                        - (2 * np.power(yt_val, 2) * np.power(yb_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((4 / 5) * np.power(g1_val, 2)
                           * np.power(yt_val, 2))
                        + (np.power(yb_val, 2)
                           * ((6 * np.power(g2_val, 2))
                              + ((4 / 5) * np.power(g1_val, 2))))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))

        dys_dt_2l = (ys_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4)
                                 + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(yc_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (3 * np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(ys_val, 4))
                        - (2 * np.power(yc_val, 4))
                        - (2 * np.power(yc_val, 2) * np.power(ys_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((4 / 5) * np.power(g1_val, 2) * np.power(yc_val, 2))
                        + (np.power(ys_val, 2)
                           * ((6 * np.power(g2_val, 2))
                              + ((4 / 5) * np.power(g1_val, 2))))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))

        dyd_dt_2l = (yd_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(yu_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (3 * np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(yd_val, 4))
                        - (2 * np.power(yu_val, 4))
                        - (2 * np.power(yd_val, 2) * np.power(yu_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((4 / 5) * np.power(g1_val, 2) * np.power(yu_val, 2))
                        + (np.power(yd_val, 2)
                           * ((6 * np.power(g2_val, 2))
                              + ((4 / 5) * np.power(g1_val, 2))))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))

        dytau_dt_2l = (ytau_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                       * (((-3) * ((3 * (np.power(yb_val, 4)
                                         + np.power(ys_val, 4)
                                         + np.power(yd_val, 4)))
                                   + (np.power(yt_val, 2)
                                      * np.power(yb_val, 2))
                                   + (np.power(yc_val, 2)
                                      * np.power(ys_val, 2))
                                   + (np.power(yu_val, 2)
                                      * np.power(yd_val, 2))
                                   + np.power(ytau_val, 4)
                                   + np.power(ymu_val, 4)
                                   + np.power(ye_val, 4)))# end trace
                          - (3 * np.power(ytau_val, 2)# Tr(3Yd^2 + Ye^2)
                             * ((3 * (np.power(yb_val, 2)
                                      + np.power(ys_val, 2)
                                      + np.power(yd_val, 2)))
                                + np.power(ytau_val, 2)
                                + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))# end trace
                          - (4 * np.power(ytau_val, 4))
                          + (((16 * np.power(g3_val, 2))
                              - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                             * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))# end trace
                          + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                             * (np.power(ytau_val, 2)
                                + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))# end trace
                          + (6 * np.power(g2_val, 2) * np.power(ytau_val, 2))
                          + ((15 / 2) * np.power(g2_val, 4))
                          + ((9 / 5) * np.power(g2_val, 2)
                             * np.power(g1_val, 2))
                          + ((27 / 2) * np.power(g1_val, 4))))

        dymu_dt_2l = (ymu_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + (np.power(yt_val, 2)
                                     * np.power(yb_val, 2))
                                  + (np.power(yc_val, 2)
                                     * np.power(ys_val, 2))
                                  + (np.power(yu_val, 2)
                                     * np.power(yd_val, 2))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(ymu_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2)
                               + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(ymu_val, 4))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2)
                               + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (6 * np.power(g2_val, 2) * np.power(ymu_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + ((9 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((27 / 2) * np.power(g1_val, 4))))

        dye_dt_2l = (ye_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2)
                                    * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2)
                                    * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2)
                                    * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4)
                                 + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(ye_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(ye_val, 4))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))
                        + (6 * np.power(g2_val, 2) * np.power(ye_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + ((9 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((27 / 2) * np.power(g1_val, 4))))

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
        dat_dt_1l = ((at_val# Tr(Yu^2)
                      * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (5 * np.power(yt_val, 2)) + np.power(yb_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((13 / 15) * np.power(g1_val, 2))))
                     + (yt_val# Tr(au*Yu)
                        * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (4 * yt_val * at_val)
                           + (2 * yb_val * ab_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((26 / 15) * np.power(g1_val, 2) * M1_val))))

        dac_dt_1l = ((ac_val# Tr(Yu^2)
                      * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (5 * np.power(yc_val, 2)) + np.power(ys_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((13 / 15) * np.power(g1_val, 2))))
                     + (yc_val# Tr(au*Yu)
                        * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (4 * yc_val * ac_val)
                           + (2 * ys_val * as_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((26 / 15) * np.power(g1_val, 2) * M1_val))))

        dau_dt_1l = ((au_val# Tr(Yu^2)
                      * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (5 * np.power(yu_val, 2)) + np.power(yd_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((13 / 15) * np.power(g1_val, 2))))
                     + (yu_val# Tr(au*Yu)
                        * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (4 * yu_val * au_val)
                           + (2 * yd_val * ad_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((26 / 15) * np.power(g1_val, 2) * M1_val))))

        dab_dt_1l = ((ab_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(yb_val, 2)) + np.power(yt_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((7 / 15) * np.power(g1_val, 2))))
                     + (yb_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                           + (4 * yb_val * ab_val) + (2 * yt_val * at_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((14 / 15) * np.power(g1_val, 2) * M1_val))))

        das_dt_1l = ((as_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(ys_val, 2)) + np.power(yc_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((7 / 15) * np.power(g1_val, 2))))
                     + (ys_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))
                           + (4 * ys_val * as_val) + (2 * yc_val * ac_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((14 / 15) * np.power(g1_val, 2) * M1_val))))

        dad_dt_1l = ((ad_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(yd_val, 2)) + np.power(yu_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((7 / 15) * np.power(g1_val, 2))))
                     + (yd_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                           + (4 * yd_val * ad_val) + (2 * yu_val * au_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((14 / 15) * np.power(g1_val, 2) * M1_val))))

        datau_dt_1l = ((atau_val# Tr(3Yd^2 + Ye^2)
                        * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                  + np.power(yd_val, 2)))
                            + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                           + (5 * np.power(ytau_val, 2))
                           - (3 * np.power(g2_val, 2))
                           - ((9 / 5) * np.power(g1_val, 2))))
                       + (ytau_val# Tr(6ad*Yd + 2ae*Ye)
                          * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                   + (ad_val * yd_val)))
                             + (2 * ((atau_val * ytau_val)
                                     + (amu_val * ymu_val)
                                     + (ae_val * ye_val)))# end trace
                             + (4 * ytau_val * atau_val)
                             + (6 * np.power(g2_val, 2) * M2_val)
                             + ((18 / 5) * np.power(g1_val, 2) * M1_val))))

        damu_dt_1l = ((amu_val# Tr(3Yd^2 + Ye^2)
                       * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))
                           + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                          + (5 * np.power(ymu_val, 2))
                          - (3 * np.power(g2_val, 2))
                          - ((9 / 5) * np.power(g1_val, 2))))
                      + (ymu_val# Tr(6ad*Yd + 2ae*Ye)
                         * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                  + (ad_val * yd_val)))
                            + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val)))# end trace
                            + (4 * ymu_val * amu_val)
                            + (6 * np.power(g2_val, 2) * M2_val)
                            + ((18 / 5) * np.power(g1_val, 2) * M1_val))))

        dae_dt_1l = ((ae_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(ye_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((9 / 5) * np.power(g1_val, 2))))
                     + (ye_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                           + (4 * ye_val * ae_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((18 / 5) * np.power(g1_val, 2) * M1_val))))

        # 2 loop parts
        dat_dt_2l = ((at_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                      * (((-3) * ((3 * (np.power(yt_val, 4)
                                        + np.power(yc_val, 4)
                                        + np.power(yu_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))))# end trace
                         - (np.power(yb_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (15 * np.power(yt_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (6 * np.power(yt_val, 4))
                         - (2 * np.power(yb_val, 4))
                         - (4 * np.power(yb_val, 2) * np.power(yt_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
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
                         + ((2743 / 450) * np.power(g1_val, 4))))
                     + (yt_val# Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                        * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                          + (ac_val * np.power(yc_val, 3))
                                          + (au_val * np.power(yu_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)))# end trace
                           - (18 * np.power(yt_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (np.power(yb_val, 2)# Tr(6ad*Yd + 2ae*Ye)
                              * ((6
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + (2
                                    * ((atau_val * ytau_val)
                                       + (amu_val * ymu_val)
                                       + (ae_val * ye_val)))))# end trace
                           - (12 * yt_val * at_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (yb_val * ab_val# Tr(6Yd^2 + 2Ye^2)
                              * ((6 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (2 * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yt_val, 3) * at_val)
                           - (8 * np.power(yb_val, 3) * ab_val)
                           - (2 * np.power(yb_val, 2) * yt_val * at_val)
                           - (4 * yb_val * ab_val * np.power(yt_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                              * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yt_val * at_val)
                           + ((4 / 5) * np.power(g1_val, 2) * yb_val * ab_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yt_val, 2))
                           - ((4 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yb_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((272 / 45) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((5486 / 225) * np.power(g1_val, 4) * M1_val))))

        dac_dt_2l = ((ac_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                      * (((-3) * ((3 * (np.power(yt_val, 4)
                                        + np.power(yc_val, 4)
                                        + np.power(yu_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))))# end trace
                         - (np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (15 * np.power(yc_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (6 * np.power(yc_val, 4))
                         - (2 * np.power(ys_val, 4))
                         - (4 * np.power(ys_val, 2) * np.power(yc_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
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
                         + ((2743 / 450) * np.power(g1_val, 4))))
                     + (yc_val# Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                        * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                          + (ac_val * np.power(yc_val, 3))
                                          + (au_val * np.power(yu_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)))# end trace
                           - (18 * np.power(yc_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (np.power(ys_val, 2)# Tr(6ad*Yd + 2ae*Ye)
                              * ((6
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + (2
                                    * ((atau_val * ytau_val)
                                       + (amu_val * ymu_val)
                                       + (ae_val * ye_val)))))# end trace
                           - (12 * yc_val * ac_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (ys_val * as_val# Tr(6Yd^2 + 2Ye^2)
                              * ((6 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (2 * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yc_val, 3) * ac_val)
                           - (8 * np.power(ys_val, 3) * as_val)
                           - (2 * np.power(ys_val, 2) * yc_val * ac_val)
                           - (4 * ys_val * as_val * np.power(yc_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                              * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yc_val * ac_val)
                           + ((4 / 5) * np.power(g1_val, 2) * ys_val * as_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yc_val, 2))
                           - ((4 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(ys_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((272 / 45) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((5486 / 225) * np.power(g1_val, 4) * M1_val))))

        dau_dt_2l = ((au_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                      * (((-3) * ((3 * (np.power(yt_val, 4)
                                        + np.power(yc_val, 4)
                                        + np.power(yu_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))))# end trace
                         - (np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (15 * np.power(yu_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (6 * np.power(yu_val, 4))
                         - (2 * np.power(yd_val, 4))
                         - (4 * np.power(yd_val, 2) * np.power(yu_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
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
                         + ((2743 / 450) * np.power(g1_val, 4))))
                     + (yu_val# Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                        * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                          + (ac_val * np.power(yc_val, 3))
                                          + (au_val * np.power(yu_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)))# end trace
                           - (18 * np.power(yu_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (np.power(yd_val, 2)# Tr(6ad*Yd + 2ae*Ye)
                              * ((6
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + (2
                                    * ((atau_val * ytau_val)
                                       + (amu_val * ymu_val)
                                       + (ae_val * ye_val)))))# end trace
                           - (12 * yu_val * au_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (yd_val * ad_val# Tr(6Yd^2 + 2Ye^2)
                              * ((6 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (2 * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yu_val, 3) * au_val)
                           - (8 * np.power(yd_val, 3) * ad_val)
                           - (2 * np.power(yd_val, 2) * yu_val * au_val)
                           - (4 * yd_val * ad_val * np.power(yu_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                              * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yu_val * au_val)
                           + ((4 / 5) * np.power(g1_val, 2) * yd_val * ad_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yu_val, 2))
                           - ((4 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yd_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((272 / 45) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((5486 / 225) * np.power(g1_val, 4) * M1_val))))

        dab_dt_2l = ((ab_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
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
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(yt_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (5 * np.power(yb_val, 2)# Tr(3Yd^2+Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(yb_val, 4))
                         - (2 * np.power(yt_val, 4))
                         - (4 * np.power(yb_val, 2) * np.power(yt_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
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
                         + ((287 / 90) * np.power(g1_val, 4))))
                     + (yb_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
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
                                            + (ae_val * np.power(ye_val, 3)))
                                       )))# end trace
                           - (6 * np.power(yt_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (6 * np.power(yb_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))# end trace
                           - (6 * yt_val * at_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (4 * yb_val * ab_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yb_val, 3) * ab_val)
                           - (8 * np.power(yt_val, 3) * at_val)
                           - (4 * np.power(yb_val, 2) * yt_val * at_val)
                           - (2 * yb_val * ab_val * np.power(yt_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + ((8 / 5) * np.power(g1_val, 2) * yt_val * at_val)
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yb_val * ab_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yb_val, 2))
                           - ((8 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yt_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((16 / 9) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((574 / 45) * np.power(g1_val, 4) * M1_val))))

        das_dt_2l = ((as_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
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
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(yc_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (5 * np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(ys_val, 4))
                         - (2 * np.power(yc_val, 4))
                         - (4 * np.power(ys_val, 2) * np.power(yc_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
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
                         + ((287 / 90) * np.power(g1_val, 4))))
                     + (ys_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
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
                                            + (ae_val * np.power(ye_val, 3)))
                                       )))# end trace
                           - (6 * np.power(yc_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (6 * np.power(ys_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))# end trace
                           - (6 * yc_val * ac_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (4 * ys_val * as_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(ys_val, 3) * as_val)
                           - (8 * np.power(yc_val, 3) * ac_val)
                           - (4 * np.power(ys_val, 2) * yc_val * ac_val)
                           - (2 * ys_val * as_val * np.power(yc_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + ((8 / 5) * np.power(g1_val, 2) * yc_val * ac_val)
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * ys_val * as_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(ys_val, 2))
                           - ((8 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yc_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((16 / 9) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((574 / 45) * np.power(g1_val, 4) * M1_val))))

        dad_dt_2l = ((ad_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
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
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(yu_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (5 * np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(yd_val, 4))
                         - (2 * np.power(yu_val, 4))
                         - (4 * np.power(yd_val, 2) * np.power(yu_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
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
                         + ((287 / 90) * np.power(g1_val, 4))))
                     + (yd_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
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
                                            + (ae_val * np.power(ye_val, 3)))
                                       )))# end trace
                           - (6 * np.power(yu_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (6 * np.power(yd_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))# end trace
                           - (6 * yu_val * au_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (4 * yd_val * ad_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yd_val, 3) * ad_val)
                           - (8 * np.power(yu_val, 3) * au_val)
                           - (4 * np.power(yd_val, 2) * yu_val * au_val)
                           - (2 * yd_val * ad_val * np.power(yu_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + ((8 / 5) * np.power(g1_val, 2) * yu_val * au_val)
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yd_val * ad_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yd_val, 2))
                           - ((8 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yu_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((16 / 9) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((574 / 45) * np.power(g1_val, 4) * M1_val))))

        datau_dt_2l = ((atau_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
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
                                    + np.power(ye_val, 4)))# end trace
                           - (5 * np.power(ytau_val, 2)# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (np.power(ytau_val, 2)
                                    + np.power(ymu_val, 2)
                                    + np.power(ye_val, 2))))# end trace
                           - (6 * np.power(ytau_val, 4))
                           + (((16 * np.power(g3_val, 2))
                               - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                              * (np.power(yb_val, 2)
                                 + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           + (((12 * np.power(g2_val, 2))
                               - ((6 / 5) * np.power(g1_val, 2)))
                              * np.power(ytau_val, 2))
                           + ((15 / 2) * np.power(g2_val, 4))
                           + ((9 / 5) * np.power(g2_val, 2)
                              * np.power(g1_val, 2))
                           + ((27 / 2) * np.power(g1_val, 4))))
                       + (ytau_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                          * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                            + (as_val * np.power(ys_val, 3))
                                            + (ad_val * np.power(yd_val, 3))))
                                      + (at_val * np.power(yb_val, 2) * yt_val)
                                      + (ac_val * np.power(ys_val, 2) * yc_val)
                                      + (au_val * np.power(yd_val, 2) * yu_val)
                                      + (ab_val * np.power(yt_val, 2) * yb_val)
                                      + (as_val * np.power(yc_val, 2) * ys_val)
                                      + (ad_val * np.power(yu_val, 2) * yd_val)
                                      + (2 * ((atau_val
                                               * np.power(ytau_val, 3))
                                              + (amu_val
                                                 * np.power(ymu_val, 3))
                                              + (ae_val
                                                 * np.power(ye_val, 3)))
                                         )))# end trace
                             - (4 * ytau_val * atau_val# Tr(3Yd^2 + Ye^2)
                                * ((3 * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))
                                   + ((np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                             - (6 * np.power(ytau_val, 2)# Tr(3ad*Yd + ae*Ye)
                                * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                                         + (ad_val * yd_val)))
                                   + (atau_val * ytau_val)
                                   + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                             - (14 * np.power(ytau_val, 3) * atau_val)
                             + (((32 * np.power(g3_val, 2))
                                 - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                                * ((ab_val * yb_val) + (as_val * ys_val)
                                   + (ad_val * yd_val)))# end trace
                             + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                                * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                             + (((6 * np.power(g2_val, 2))
                                 + ((6 / 5) * np.power(g1_val, 2)))
                                * ytau_val * atau_val)
                             - (((32 * np.power(g3_val, 2) * M3_val)
                                 - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                                * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                   + np.power(yd_val, 2)))# end trace
                             - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                                * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))# end trace
                             - (12 * np.power(g2_val, 2) * M2_val
                                * np.power(ytau_val, 2))
                             - (30 * np.power(g2_val, 4) * M2_val)
                             - ((18 / 5) * np.power(g2_val, 2)
                                * np.power(g1_val, 2)
                                * (M1_val + M2_val))
                             - (54 * np.power(g1_val, 4) * M1_val))))

        damu_dt_2l = ((amu_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
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
                                   + np.power(ye_val, 4)))# end trace
                          - (5 * np.power(ymu_val, 2)# Tr(3Yd^2 + Ye^2)
                             * ((3 * (np.power(yb_val, 2)
                                      + np.power(ys_val, 2)
                                      + np.power(yd_val, 2)))
                                + (np.power(ytau_val, 2)
                                   + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2))))# end trace
                          - (6 * np.power(ymu_val, 4))
                          + (((16 * np.power(g3_val, 2))
                              - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                             * (np.power(yb_val, 2)
                                + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))# end trace
                          + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                             * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))# end trace
                          + (((12 * np.power(g2_val, 2))
                              - ((6 / 5) * np.power(g1_val, 2)))
                             * np.power(ymu_val, 2))
                          + ((15 / 2) * np.power(g2_val, 4))
                          + ((9 / 5) * np.power(g2_val, 2)
                             * np.power(g1_val, 2))
                          + ((27 / 2) * np.power(g1_val, 4))))
                      + (ymu_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
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
                                             + (ae_val * np.power(ye_val, 3))))))# end trace
                            - (4 * ymu_val * amu_val# Tr(3Yd^2 + Ye^2)
                               * ((3 * (np.power(yb_val, 2)
                                        + np.power(ys_val, 2)
                                        + np.power(yd_val, 2)))
                                  + ((np.power(ytau_val, 2)
                                      + np.power(ymu_val, 2)
                                      + np.power(ye_val, 2)))))# end trace
                            - (6 * np.power(ymu_val, 2)# Tr(3ad*Yd + ae*Ye)
                               * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                                        + (ad_val * yd_val)))
                                  + (atau_val * ytau_val) + (amu_val * ymu_val)
                                  + (ae_val * ye_val)))# end trace
                            - (14 * np.power(ymu_val, 3) * amu_val)
                            + (((32 * np.power(g3_val, 2))
                                - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                               * ((ab_val * yb_val) + (as_val * ys_val)
                                  + (ad_val * yd_val)))# end trace
                            + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                               * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                  + (ae_val * ye_val)))# end trace
                            + (((6 * np.power(g2_val, 2))
                                + ((6 / 5) * np.power(g1_val, 2)))
                               * ymu_val * amu_val)
                            - (((32 * np.power(g3_val, 2) * M3_val)
                                - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                               * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                  + np.power(yd_val, 2)))# end trace
                            - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                               * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2)))# end trace
                            - (12 * np.power(g2_val, 2) * M2_val
                               * np.power(ymu_val, 2))
                            - (30 * np.power(g2_val, 4) * M2_val)
                            - ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                               * (M1_val + M2_val))
                            - (54 * np.power(g1_val, 4) * M1_val))))

        dae_dt_2l = ((ae_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
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
                                  + np.power(ye_val, 4)))# end trace
                         - (5 * np.power(ye_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(ye_val, 4))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((12 * np.power(g2_val, 2))
                             - ((6 / 5) * np.power(g1_val, 2)))
                            * np.power(ye_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + ((9 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2))
                         + ((27 / 2) * np.power(g1_val, 4))))
                     + (ye_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
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
                                            + (ae_val * np.power(ye_val, 3))))))# end trace
                           - (4 * ye_val * ae_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (6 * np.power(ye_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                                       + (ad_val * yd_val)))
                                 + (atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           - (14 * np.power(ye_val, 3) * ae_val)
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * ye_val * ae_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (12 * np.power(g2_val, 2) * M2_val
                              * np.power(ye_val, 2))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M1_val + M2_val))
                           - (54 * np.power(g1_val, 4) * M1_val))))

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
        # db_dt_1l = ((b_val# Tr(3Yu^2 + 3Yd^2 + Ye^2)
        #             * (((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
        #                       + np.power(yu_val, 2) + np.power(yb_val, 2)
        #                       + np.power(ys_val, 2) + np.power(yd_val, 2)))
        #                 + np.power(ytau_val, 2) + np.power(ymu_val, 2)
        #                 + np.power(ye_val, 2))# end trace
        #                - (3 * np.power(g2_val, 2))
        #                - ((3 / 5) * np.power(g1_val, 2))))
        #            + (mu_val# Tr(6au*Yu + 6ad*Yd + 2ae*Ye)
        #               * (((6 * ((at_val * yt_val) + (ac_val * yc_val)
        #                         + (au_val * yu_val) + (ab_val * yb_val)
        #                         + (as_val * ys_val) + (ad_val * yd_val)))
        #                   + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
        #                           + (ae_val * ye_val))))
        #                  + (6 * np.power(g2_val, 2) * M2_val)
        #                  + ((6 / 5) * np.power(g1_val, 2) * M1_val))))

        # 2 loop part
        # db_dt_2l = ((b_val# Tr(3Yu^4 + 3Yd^4 + 2Yu^2*Yd^2 + Ye^4)
        #             * (((-3) * ((3 * (np.power(yt_val, 4) + np.power(yc_val, 4)
        #                               + np.power(yu_val, 4)
        #                               + np.power(yb_val, 4)
        #                               + np.power(ys_val, 4)
        #                               + np.power(yd_val, 4)))
        #                         + (2 * ((np.power(yt_val, 2)
        #                                  * np.power(yb_val, 2))
        #                                 + (np.power(yc_val, 2)
        #                                    * np.power(ys_val, 2))
        #                                 + (np.power(yu_val, 2)
        #                                    * np.power(yd_val, 2))))
        #                         + np.power(ytau_val, 4) + np.power(ymu_val, 4)
        #                         + np.power(ye_val, 4)))# end trace
        #                + (((16 * np.power(g3_val, 2))
        #                    + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
        #                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
        #                      + np.power(yu_val, 2)))# end trace
        #                + (((16 * np.power(g3_val, 2))
        #                    - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
        #                   * (np.power(yb_val, 2) + np.power(ys_val, 2)
        #                      + np.power(yd_val, 2)))# end trace
        #                + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
        #                   * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
        #                      + np.power(ye_val, 2)))# end trace
        #                + ((15 / 2) * np.power(g2_val, 4))
        #                + ((9 / 5) * np.power(g1_val, 2) * np.power(g2_val, 2))
        #                + ((207 / 50) * np.power(g1_val, 4))))
        #            + (mu_val * (((-12)# Tr(3au*Yu^3 + 3ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + ae*Ye^3)
        #                  * ((3 * ((at_val * np.power(yt_val, 3))
        #                           + (ac_val * np.power(yc_val, 3))
        #                           + (au_val * np.power(yu_val, 3))
        #                           + (ab_val * np.power(yb_val, 3))
        #                           + (as_val * np.power(ys_val, 3))
        #                           + (ad_val * np.power(yd_val, 3))))
        #                     + ((at_val * np.power(yb_val, 2) * yt_val)
        #                        + (ac_val * np.power(ys_val, 2) * yc_val)
        #                        + (au_val * np.power(yd_val, 2) * yu_val))
        #                     + ((ab_val * np.power(yt_val, 2) * yb_val)
        #                        + (as_val * np.power(yc_val, 2) * ys_val)
        #                        + (ad_val * np.power(yu_val, 2) * yd_val))
        #                     + ((atau_val * np.power(ytau_val, 3))
        #                        + (amu_val * np.power(ymu_val, 3))
        #                        + (ae_val * np.power(ye_val, 3)))))# end trace
        #                + (((32 * np.power(g3_val, 2))
        #                     + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
        #                    * ((at_val * yt_val) + (ac_val * yc_val)
        #                       + (au_val * yu_val)))# end trace
        #                 + (((32 * np.power(g3_val, 2))
        #                     - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
        #                    * ((ab_val * yb_val) + (as_val * ys_val)
        #                       + (ad_val * yd_val)))# end trace
        #                 + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
        #                    * ((atau_val * ytau_val) + (amu_val * ymu_val)
        #                       + (ae_val * ye_val)))# end trace
        #                 - (((32 * np.power(g3_val, 2) * M3_val)
        #                     + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
        #                    * (np.power(yt_val, 2) + np.power(yc_val, 2)
        #                       + np.power(yu_val, 2)))
        #                 - (((32 * np.power(g3_val, 2) * M3_val)# end trace
        #                     - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
        #                    * (np.power(yb_val, 2) + np.power(ys_val, 2)
        #                       + np.power(yd_val, 2)))# end trace
        #                 - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
        #                    * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
        #                       + np.power(ye_val, 2)))# end trace
        #                 - (30 * np.power(g2_val, 4) * M2_val)
        #                 - ((18 / 5) * np.power(g1_val, 2)
        #                    * np.power(g2_val, 2)
        #                    * (M1_val + M2_val))
        #                 - ((414 / 25) * np.power(g1_val, 4) * M1_val))))

        # Total b beta function
        # db_dt = (1 / t) * ((loop_fac * db_dt_1l)
        #                   + (loop_fac_sq * db_dt_2l))

        ##### Scalar squared masses #####
        # Introduce S, S', and sigma terms
        S_val = (mHu_sq_val - mHd_sq_val + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
                 - mL3_sq_val - mL2_sq_val - mL1_sq_val
                 - (2 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
                 + mD3_sq_val + mD2_sq_val + mD1_sq_val
                 + mE3_sq_val + mE2_sq_val + mE1_sq_val)

        # Tr(-(3mHu^2 + mQ^2) * Yu^2 + 4Yu^2 * mU^2 + (3mHd^2 - mQ^2) * Yd^2
        #    - 2Yd^2 * mD^2 + (mHd^2 + mL^2) * Ye^2 - 2Ye^2 * mE^2)
        Spr_val = ((((-1) * ((((3 * mHu_sq_val) + mQ3_sq_val)
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
                            + (np.power(ye_val, 2) * mE1_sq_val))))# end trace
                   + ((((3 / 2) * np.power(g2_val, 2))
                       + ((3 / 10) * np.power(g1_val, 2)))
                      * (mHu_sq_val - mHd_sq_val# Tr(mL^2)
                         - (mL3_sq_val + mL2_sq_val + mL1_sq_val)))# end trace
                   + ((((8 / 3) * np.power(g3_val, 2))
                       + ((3 / 2) * np.power(g2_val, 2))
                       + ((1 / 30) * np.power(g1_val, 2)))# Tr(mQ^2)
                      * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))# end trace
                   - ((((16 / 3) * np.power(g3_val, 2))
                       + ((16 / 15) * np.power(g1_val, 2)))# Tr (mU^2)
                      * (mU3_sq_val + mU2_sq_val + mU1_sq_val))# end trace
                   + ((((8 / 3) * np.power(g3_val, 2))
                      + ((2 / 15) * np.power(g1_val, 2)))# Tr(mD^2)
                      * (mD3_sq_val + mD2_sq_val + mD1_sq_val))# end trace
                   + ((6 / 5) * np.power(g1_val, 2)# Tr(mE^2)
                      * (mE3_sq_val + mE2_sq_val + mE1_sq_val)))# end trace

        sigma1 = ((1 / 5) * np.power(g1_val, 2)
                  * ((3 * (mHu_sq_val + mHd_sq_val))# Tr(mQ^2 + 3mL^2 + 8mU^2 + 2mD^2 + 6mE^2)
                     + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
                     + (3 * (mL3_sq_val + mL2_sq_val + mL1_sq_val))
                     + (8 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
                     + (2 * (mD3_sq_val + mD2_sq_val + mD1_sq_val))
                     + (6 * (mE3_sq_val + mE2_sq_val + mE1_sq_val))))# end trace

        sigma2 = (np.power(g2_val, 2)
                  * (mHu_sq_val + mHd_sq_val# Tr(3mQ^2 + mL^2)
                     + (3 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
                     + mL3_sq_val + mL2_sq_val + mL1_sq_val))# end trace

        sigma3 = (np.power(g3_val, 2)# Tr(2mQ^2 + mU^2 + mD^2)
                  * ((2 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
                     + mU3_sq_val + mU2_sq_val + mU1_sq_val
                     + mD3_sq_val + mD2_sq_val + mD1_sq_val))# end trace

        # 1 loop part of Higgs squared masses
        dmHu_sq_dt_1l = ((6# Tr((mHu^2 + mQ^2) * Yu^2 + Yu^2 * mU^2 + au^2)
                          * (((mHu_sq_val + mQ3_sq_val) * np.power(yt_val, 2))
                             + ((mHu_sq_val + mQ2_sq_val)
                                * np.power(yc_val, 2))
                             + ((mHu_sq_val + mQ1_sq_val)
                                * np.power(yu_val, 2))
                             + (mU3_sq_val * np.power(yt_val, 2))
                             + (mU2_sq_val * np.power(yc_val, 2))
                             + (mU1_sq_val * np.power(yu_val, 2))
                             + np.power(at_val, 2) + np.power(ac_val, 2)
                             + np.power(au_val, 2)))# end trace
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((3 / 5) * np.power(g1_val, 2) * S_val))

        # Tr (6(mHd^2 + mQ^2) * Yd^2 + 6Yd^2*mD^2 + 2(mHd^2 + mL^2) * Ye^2
        #     + 2(Ye^2 * mE^2) + 6ad^2 + 2ae^2)
        dmHd_sq_dt_1l = ((6 * (((mHd_sq_val + mQ3_sq_val)
                                * np.power(yb_val, 2))
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
                                  + np.power(ae_val, 2))))# end trace
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        # 2 loop part of Higgs squared masses
        dmHu_sq_dt_2l = (((-6) # Tr(6(mHu^2 + mQ^2)*Yu^4 + 6Yu^4 * mU^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2 * Yd^2 + Yu^2 * Yd^2 * mU^2 + Yu^2 * Yd^2 * mQ^2 + Yu^2 * Yd^2 * mD^2 + 12au^2 * Yu^2 + ad^2 * Yu^2 + Yd^2 * au^2 + 2ad * Yd * Yu * au)
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
                                     + (yd_val * ad_val * au_val * yu_val)))))# end trace
                         + (((32 * np.power(g3_val, 2))
                             + ((8 / 5) * np.power(g1_val, 2))) # Tr((mHu^2 + mQ^2 + mU^2) * Yu^2 + au^2)
                            * (((mHu_sq_val + mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mHu_sq_val + mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mHu_sq_val + mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))
                               + np.power(at_val, 2) + np.power(ac_val, 2)
                               + np.power(au_val, 2)))# end trace
                         + (32 * np.power(g3_val, 2)
                            * ((2 * np.power(M3_val, 2)# Tr(Yu^2)
                                * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                               - (2 * M3_val# Tr(Yu*au)
                                  * ((yt_val * at_val) + (yc_val * ac_val)
                                     + (yu_val * au_val)))))# end trace
                         + ((8 / 5) * np.power(g1_val, 2)
                            * ((2 * np.power(M1_val, 2)# Tr(Yu^2)
                                * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                               - (2 * M1_val# Tr(Yu*au)
                                  * ((yt_val * at_val) + (yc_val * ac_val)
                                     + (yu_val * au_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M2_val, 2) + np.power(M1_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        dmHd_sq_dt_2l = (((-6) # Tr(6(mHd^2 + mQ^2)*Yd^4 + 6Yd^4 * mD^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2 * Yd^2 + Yu^2 * Yd^2 * mU^2 + Yu^2 * Yd^2 * mQ^2 + Yu^2 * Yd^2 * mD^2 + 2(mHd^2 + mL^2) * Ye^4 + 2Ye^4 * mE^2 + 12ad^2 * Yd^2 + ad^2 * Yu^2 + Yd^2 * au^2 + 2ad * Yd * Yu * au + 4ae^2 * Ye^2)
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
                                                * np.power(ye_val, 2))))))))# end trace
                         + (((32 * np.power(g3_val, 2))
                             - ((4 / 5) * np.power(g1_val, 2))) # Tr((mHd^2 + mQ^2 + mD^2) * Yd^2 + ad^2)
                            * (((mHu_sq_val + mQ3_sq_val + mD3_sq_val)
                                * np.power(yb_val, 2))
                               + ((mHu_sq_val + mQ2_sq_val + mD2_sq_val)
                                  * np.power(ys_val, 2))
                               + ((mHu_sq_val + mQ1_sq_val + mD1_sq_val)
                                  * np.power(yd_val, 2))
                               + np.power(ab_val, 2) + np.power(as_val, 2)
                               + np.power(ad_val, 2)))# end trace
                         + (32 * np.power(g3_val, 2)
                            * ((2 * np.power(M3_val, 2)# Tr(Yd^2)
                                * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                   + np.power(yd_val, 2)))# end trace
                               - (2 * M3_val # Tr(Yd*ad)
                                  * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))))# end trace
                         - ((4 / 5) * np.power(g1_val, 2)
                            * ((2 * np.power(M1_val, 2)# Tr(Yd^2)
                                * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                   + np.power(yd_val, 2)))# end trace
                               - (2 * M1_val # Tr(Yd*ad)
                                  * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))))# end trace
                         + ((12 / 5) * np.power(g1_val, 2)
                            * (# Tr((mHd^2 + mL^2 + mE^2) * Ye^2 + ae^2)
                               ((mHd_sq_val + mL3_sq_val + mE3_sq_val)
                                * np.power(ytau_val, 2))
                               + ((mHd_sq_val + mL2_sq_val + mE2_sq_val)
                                  * np.power(ymu_val, 2))
                               + ((mHd_sq_val + mL1_sq_val + mE1_sq_val)
                                  * np.power(ye_val, 2))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M2_val, 2) + np.power(M1_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        # Total Higgs squared mass beta functions
        dmHu_sq_dt = (1 / t) * ((loop_fac * dmHu_sq_dt_1l)
                                + (loop_fac_sq * dmHu_sq_dt_2l))

        dmHd_sq_dt = (1 / t) * ((loop_fac * dmHd_sq_dt_1l)
                                + (loop_fac_sq * dmHd_sq_dt_2l))

        # 1 loop parts of scalar squared masses
        # Left squarks
        dmQ3_sq_dt_1l = (((mQ3_sq_val + (2 * mHu_sq_val))
                          * np.power(yt_val, 2))
                         + ((mQ3_sq_val + (2 * mHd_sq_val))
                            * np.power(yb_val, 2))
                         + ((np.power(yt_val, 2) + np.power(yb_val, 2))
                            * mQ3_sq_val)
                         + (2 * np.power(yt_val, 2) * mU3_sq_val)
                         + (2 * np.power(yb_val, 2) * mD3_sq_val)
                         + (2 * np.power(at_val, 2))
                         + (2 * np.power(ab_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((2 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((1 / 5) * np.power(g1_val, 2) * S_val))

        dmQ2_sq_dt_1l = (((mQ2_sq_val + (2 * mHu_sq_val))
                          * np.power(yc_val, 2))
                         + ((mQ2_sq_val + (2 * mHd_sq_val))
                            * np.power(ys_val, 2))
                         + ((np.power(yc_val, 2) + np.power(ys_val, 2))
                            * mQ2_sq_val)
                         + (2 * np.power(yc_val, 2) * mU2_sq_val)
                         + (2 * np.power(ys_val, 2) * mD2_sq_val)
                         + (2 * np.power(ac_val, 2))
                         + (2 * np.power(as_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((2 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((1 / 5) * np.power(g1_val, 2) * S_val))

        dmQ1_sq_dt_1l = (((mQ1_sq_val + (2 * mHu_sq_val))
                          * np.power(yu_val, 2))
                         + ((mQ1_sq_val + (2 * mHd_sq_val))
                            * np.power(yd_val, 2))
                         + ((np.power(yu_val, 2)
                             + np.power(yd_val, 2)) * mQ1_sq_val)
                         + (2 * np.power(yu_val, 2) * mU1_sq_val)
                         + (2 * np.power(yd_val, 2) * mD1_sq_val)
                         + (2 * np.power(au_val, 2))
                         + (2 * np.power(ad_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((2 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((1 / 5) * np.power(g1_val, 2) * S_val))

        # Left leptons
        dmL3_sq_dt_1l = (((mL3_sq_val + (2 * mHd_sq_val))
                          * np.power(ytau_val, 2))
                         + (2 * np.power(ytau_val, 2) * mE3_sq_val)
                         + (np.power(ytau_val, 2) * mL3_sq_val)
                         + (2 * np.power(atau_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        dmL2_sq_dt_1l = (((mL2_sq_val + (2 * mHd_sq_val))
                          * np.power(ymu_val, 2))
                         + (2 * np.power(ymu_val, 2) * mE2_sq_val)
                         + (np.power(ymu_val, 2) * mL2_sq_val)
                         + (2 * np.power(amu_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        dmL1_sq_dt_1l = (((mL1_sq_val + (2 * mHd_sq_val))
                          * np.power(ye_val, 2))
                         + (2 * np.power(ye_val, 2) * mE1_sq_val)
                         + (np.power(ye_val, 2) * mL1_sq_val)
                         + (2 * np.power(ae_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        # Right up-type squarks
        dmU3_sq_dt_1l = ((2 * (mU3_sq_val + (2 * mHd_sq_val))
                          * np.power(yt_val, 2))
                         + (4 * np.power(yt_val, 2) * mQ3_sq_val)
                         + (2 * np.power(yt_val, 2) * mU3_sq_val)
                         + (4 * np.power(at_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((32 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((4 / 5) * np.power(g1_val, 2) * S_val))

        dmU2_sq_dt_1l = ((2 * (mU2_sq_val + (2 * mHd_sq_val))
                          * np.power(yc_val, 2))
                         + (4 * np.power(yc_val, 2) * mQ2_sq_val)
                         + (2 * np.power(yc_val, 2) * mU2_sq_val)
                         + (4 * np.power(ac_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((32 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((4 / 5) * np.power(g1_val, 2) * S_val))

        dmU1_sq_dt_1l = ((2 * (mU1_sq_val + (2 * mHd_sq_val))
                          * np.power(yu_val, 2))
                         + (4 * np.power(yu_val, 2) * mQ1_sq_val)
                         + (2 * np.power(yu_val, 2) * mU1_sq_val)
                         + (4 * np.power(au_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((32 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((4 / 5) * np.power(g1_val, 2) * S_val))

        # Right down-type squarks
        dmD3_sq_dt_1l = ((2 * (mD3_sq_val + (2 * mHd_sq_val))
                          * np.power(yb_val, 2))
                         + (4 * np.power(yb_val, 2) * mQ3_sq_val)
                         + (2 * np.power(yb_val, 2) * mD3_sq_val)
                         + (4 * np.power(ab_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((8 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2) * S_val))

        dmD2_sq_dt_1l = ((2 * (mD2_sq_val + (2 * mHd_sq_val))
                          * np.power(ys_val, 2))
                         + (4 * np.power(ys_val, 2) * mQ2_sq_val)
                         + (2 * np.power(ys_val, 2) * mD2_sq_val)
                         + (4 * np.power(as_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((8 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2) * S_val))

        dmD1_sq_dt_1l = ((2 * (mD1_sq_val + (2 * mHd_sq_val))
                          * np.power(yd_val, 2))
                         + (4 * np.power(yd_val, 2) * mQ1_sq_val)
                         + (2 * np.power(yd_val, 2) * mD1_sq_val)
                         + (4 * np.power(ad_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((8 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2) * S_val))

        # Right leptons
        dmE3_sq_dt_1l = ((2 * (mE3_sq_val + (2 * mHd_sq_val))
                          * np.power(ytau_val, 2))
                         + (4 * np.power(ytau_val, 2) * mL3_sq_val)
                         + (2 * np.power(ytau_val, 2) * mE3_sq_val)
                         + (4 * np.power(atau_val, 2))
                         - ((24 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((6 / 5) * np.power(g1_val, 2) * S_val))

        dmE2_sq_dt_1l = ((2 * (mE2_sq_val + (2 * mHd_sq_val))
                          * np.power(ymu_val, 2))
                         + (4 * np.power(ymu_val, 2) * mL2_sq_val)
                         + (2 * np.power(ymu_val, 2) * mE2_sq_val)
                         + (4 * np.power(amu_val, 2))
                         - ((24 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((6 / 5) * np.power(g1_val, 2) * S_val))

        dmE1_sq_dt_1l = ((2 * (mE1_sq_val + (2 * mHd_sq_val))
                          * np.power(ye_val, 2))
                         + (4 * np.power(ye_val, 2) * mL1_sq_val)
                         + (2 * np.power(ye_val, 2) * mE1_sq_val)
                         + (4 * np.power(ae_val, 2))
                         - ((24 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((6 / 5) * np.power(g1_val, 2) * S_val))

        # 2 loop parts of scalar squared masses
        # Left squarks
        dmQ3_sq_dt_2l = (((-8) * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                          * np.power(yt_val, 4))
                         - (8 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                            * np.power(yb_val, 4))
                         - (np.power(yt_val, 2)
                            * ((2 * mQ3_sq_val) + (2 * mU3_sq_val)
                               + (4 * mHu_sq_val))# Tr(3Yu^2)
                            * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yb_val, 2)
                            * ((2 * mQ3_sq_val) + (2 * mD3_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (6 * np.power(yt_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (np.power(yb_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(yt_val, 2) * np.power(at_val, 2))
                         - (16 * np.power(yb_val, 2) * np.power(ab_val, 2))
                         - (np.power(at_val, 2)# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yt_val, 2)# Tr(6au^2)
                            * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                                   + np.power(au_val, 2)))# end trace
                         - (at_val * yt_val# Tr(12Yu*au)
                            * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val)))# end trace
                         - (np.power(ab_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(yb_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * ab_val * yb_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((2 / 5) * np.power(g1_val, 2)
                            * ((4 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + (4 * np.power(at_val, 2))
                               - (8 * M1_val * at_val * yt_val)
                               + (8 * np.power(M1_val, 2) * np.power(yt_val, 2))
                               + (2 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                                  * np.power(yb_val, 2))
                               + (2 * np.power(ab_val, 2))
                               - (4 * M1_val * ab_val * yb_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(yb_val, 2))))
                         + ((2 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
                            * (np.power(M3_val, 2) + np.power(M2_val, 2)
                               + (M2_val * M3_val)))
                         + ((32 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((1 / 15) * np.power(g1_val, 2) * sigma1))

        dmQ2_sq_dt_2l = (((-8) * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                          * np.power(yc_val, 4))
                         - (8 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                            * np.power(ys_val, 4))
                         - (np.power(yc_val, 2)
                            * ((2 * mQ2_sq_val) + (2 * mU2_sq_val)
                               + (4 * mHu_sq_val))# Tr(3Yu^2)
                            * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(ys_val, 2)
                            * ((2 * mQ2_sq_val) + (2 * mD2_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (6 * np.power(yc_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (np.power(ys_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(yc_val, 2) * np.power(ac_val, 2))
                         - (16 * np.power(ys_val, 2) * np.power(as_val, 2))
                         - (np.power(ac_val, 2)# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yc_val, 2)# Tr(6au^2)
                            * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                                   + np.power(au_val, 2)))# end trace
                         - (ac_val * yc_val# Tr(12Yu*au)
                            * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val)))# end trace
                         - (np.power(as_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ys_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * as_val * ys_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((2 / 5) * np.power(g1_val, 2)
                            * ((4 * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                                * np.power(yc_val, 2))
                               + (4 * np.power(ac_val, 2))
                               - (8 * M1_val * ac_val * yc_val)
                               + (8 * np.power(M1_val, 2) * np.power(yc_val, 2))
                               + (2 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                                  * np.power(ys_val, 2))
                               + (2 * np.power(as_val, 2))
                               - (4 * M1_val * as_val * ys_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ys_val, 2))))
                         + ((2 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
                            * (np.power(M3_val, 2) + np.power(M2_val, 2)
                               + (M2_val * M3_val)))
                         + ((32 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((1 / 15) * np.power(g1_val, 2) * sigma1))

        dmQ1_sq_dt_2l = (((-8) * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                          * np.power(yu_val, 4))
                         - (8 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                            * np.power(yd_val, 4))
                         - (np.power(yu_val, 2)
                            * ((2 * mQ1_sq_val) + (2 * mU1_sq_val)
                               + (4 * mHu_sq_val))# Tr(3Yu^2)
                            * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yd_val, 2)
                            * ((2 * mQ1_sq_val) + (2 * mD1_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (6 * np.power(yu_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (np.power(yd_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(yu_val, 2) * np.power(au_val, 2))
                         - (16 * np.power(yd_val, 2) * np.power(ad_val, 2))
                         - (np.power(au_val, 2)# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yu_val, 2)# Tr(6au^2)
                            * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                                   + np.power(au_val, 2)))# end trace
                         - (au_val * yu_val# Tr(12Yu*au)
                            * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val)))# end trace
                         - (np.power(ad_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(yd_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * ad_val * yd_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((2 / 5) * np.power(g1_val, 2)
                            * ((4 * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                                * np.power(yu_val, 2))
                               + (4 * np.power(au_val, 2))
                               - (8 * M1_val * au_val * yu_val)
                               + (8 * np.power(M1_val, 2) * np.power(yu_val, 2))
                               + (2 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                                  * np.power(yd_val, 2))
                               + (2 * np.power(ad_val, 2))
                               - (4 * M1_val * ad_val * yd_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(yd_val, 2))))
                         + ((2 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
                            * (np.power(M3_val, 2) + np.power(M2_val, 2)
                               + (M2_val * M3_val)))
                         + ((32 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((1 / 15) * np.power(g1_val, 2) * sigma1))

        # Left leptons
        dmL3_sq_dt_2l = (((-8) * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                          * np.power(ytau_val, 4))
                         - (np.power(ytau_val, 2)
                            * ((2 * mL3_sq_val) + (2 * mE3_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (np.power(ytau_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ytau_val, 2) * np.power(atau_val, 2))
                         - (np.power(atau_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ytau_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * atau_val * ytau_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)
                            * ((2 * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                                * np.power(ytau_val, 2))
                               + (2 * np.power(atau_val, 2))
                               - (4 * M1_val * atau_val
                                  * ytau_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ytau_val, 2))))
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        dmL2_sq_dt_2l = (((-8) * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                          * np.power(ymu_val, 4))
                         - (np.power(ymu_val, 2)
                            * ((2 * mL2_sq_val) + (2 * mE2_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (np.power(ymu_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ymu_val, 2) * np.power(amu_val, 2))
                         - (np.power(amu_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ymu_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * amu_val * ymu_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)
                            * ((2 * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                                * np.power(ymu_val, 2))
                               + (2 * np.power(amu_val, 2))
                               - (4 * M1_val * amu_val
                                  * ymu_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ymu_val, 2))))
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        dmL1_sq_dt_2l = (((-8) * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                          * np.power(ye_val, 4))
                         - (np.power(ye_val, 2)
                            * ((2 * mL1_sq_val) + (2 * mE1_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (np.power(ye_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ye_val, 2) * np.power(ae_val, 2))
                         - (np.power(ae_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ye_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * ae_val * ye_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)
                            * ((2 * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                                * np.power(ye_val, 2))
                               + (2 * np.power(ae_val, 2))
                               - (4 * M1_val * ae_val
                                  * ye_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ye_val, 2))))
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        # Right up-type squarks
        dmU3_sq_dt_2l = (((-8) * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                          * np.power(yt_val, 4))
                         - (4 * (mU3_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ3_sq_val) + mD3_sq_val)
                            * np.power(yb_val, 2) * np.power(yt_val, 2))
                         - (np.power(yt_val, 2)
                            * ((2 * mQ3_sq_val) + (2 * mU3_sq_val)
                               + (4 * mHu_sq_val))# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (12 * np.power(yt_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (16 * np.power(yt_val, 2) * np.power(at_val, 2))
                         - (16 * at_val * ab_val * yb_val * yt_val)
                         - (12 * ((np.power(at_val, 2)# Tr(Yu^2)
                                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                      + np.power(yu_val, 2)))# end trace
                                  + (np.power(yt_val, 2) # Tr(au^2)
                                     * (np.power(at_val, 2)
                                        + np.power(ac_val, 2)
                                        + np.power(au_val, 2)))# end trace
                                  + (at_val * yt_val * 2# Tr(Yu*au)
                                     * ((yt_val * at_val) + (yc_val * ac_val)
                                        + (yu_val * au_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + (2 * np.power(at_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yt_val, 2))
                                   - (M2_val * at_val * yt_val)))
                         - ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yt_val, 2))
                                   - (M1_val * at_val * yt_val)))
                         - ((8 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((512 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((3424 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((16 / 15) * np.power(g1_val, 2) * sigma1))

        dmU2_sq_dt_2l = (((-8) * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                          * np.power(yc_val, 4))
                         - (4 * (mU2_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ2_sq_val)
                                 + mD2_sq_val)
                            * np.power(ys_val, 2) * np.power(yc_val, 2))
                         - (np.power(yc_val, 2)
                            * ((2 * mQ2_sq_val) + (2 * mU2_sq_val)
                               + (4 * mHu_sq_val))# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (12 * np.power(yc_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (16 * np.power(yc_val, 2) * np.power(ac_val, 2))
                         - (16 * ac_val * as_val * ys_val * yc_val)
                         - (12 * ((np.power(ac_val, 2)# Tr(Yu^2)
                                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                      + np.power(yu_val, 2)))# end trace
                                  + (np.power(yc_val, 2) # Tr(au^2)
                                     * (np.power(at_val, 2)
                                        + np.power(ac_val, 2)
                                        + np.power(au_val, 2)))# end trace
                                  + (ac_val * yc_val * 2# Tr(Yu*au)
                                     * ((yt_val * at_val) + (yc_val * ac_val)
                                        + (yu_val * au_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                                * np.power(yc_val, 2))
                               + (2 * np.power(ac_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yc_val, 2))
                                   - (M2_val * ac_val * yc_val)))
                         - ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yc_val, 2))
                                   - (M1_val * ac_val * yc_val)))
                         - ((8 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((512 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((3424 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((16 / 15) * np.power(g1_val, 2) * sigma1))

        dmU1_sq_dt_2l = (((-8) * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                          * np.power(yu_val, 4))
                         - (4 * (mU1_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ1_sq_val)
                                 + mD1_sq_val)
                            * np.power(yd_val, 2) * np.power(yu_val, 2))
                         - (np.power(yu_val, 2)
                            * ((2 * mQ1_sq_val) + (2 * mU1_sq_val)
                               + (4 * mHu_sq_val))# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (12 * np.power(yu_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (16 * np.power(yu_val, 2) * np.power(au_val, 2))
                         - (16 * au_val * ad_val * yd_val * yu_val)
                         - (12 * ((np.power(au_val, 2)# Tr(Yu^2)
                                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                      + np.power(yu_val, 2)))# end trace
                                  + (np.power(yu_val, 2) # Tr(au^2)
                                     * (np.power(at_val, 2)
                                        + np.power(ac_val, 2)
                                        + np.power(au_val, 2)))# end trace
                                  + (au_val * yu_val * 2# Tr(Yu*au)
                                     * ((yt_val * at_val) + (yc_val * ac_val)
                                        + (yu_val * au_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                                * np.power(yu_val, 2))
                               + (2 * np.power(au_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yu_val, 2))
                                   - (M2_val * au_val * yu_val)))
                         - ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yu_val, 2))
                                   - (M1_val * au_val * yu_val)))
                         - ((8 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((512 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((3424 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((16 / 15) * np.power(g1_val, 2) * sigma1))

        # Right down-type squarks
        dmD3_sq_dt_2l = (((-8) * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                          * np.power(yb_val, 4))
                         - (4 * (mU3_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ3_sq_val)
                                 + mD3_sq_val) * np.power(yb_val, 2)
                            * np.power(yt_val, 2))
                         - (np.power(yb_val, 2)
                            * (2 * (mD3_sq_val + mQ3_sq_val
                                    + (2 * mHd_sq_val)))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(yb_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (((mL3_sq_val + mE3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mL2_sq_val + mE2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mL1_sq_val + mE1_sq_val)
                                     * np.power(ye_val, 2)))# end trace
                               ))
                         - (16 * np.power(yb_val, 2) * np.power(ab_val, 2))
                         - (16 * at_val * ab_val * yb_val * yt_val)
                         - (4 * np.power(ab_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(yb_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - (8 * ab_val * yb_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((6 * np.power(g2_val, 2))
                             + ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                                * np.power(yb_val, 2))
                               + (2 * np.power(ab_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yb_val, 2))
                                   - (M2_val * ab_val * yb_val)))
                         + ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yb_val, 2))
                                   - (M1_val * ab_val * yb_val)))
                         + ((4 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((128 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((808 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((4 / 15) * np.power(g1_val, 2) * sigma1))

        dmD2_sq_dt_2l = (((-8) * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                          * np.power(ys_val, 4))
                         - (4 * (mU2_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ2_sq_val)
                                 + mD2_sq_val) * np.power(ys_val, 2)
                            * np.power(yc_val, 2))
                         - (np.power(ys_val, 2)
                            * (2 * (mD2_sq_val + mQ2_sq_val
                                    + (2 * mHd_sq_val)))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ys_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (((mL3_sq_val + mE3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mL2_sq_val + mE2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mL1_sq_val + mE1_sq_val)
                                     * np.power(ye_val, 2)))# end trace
                               ))
                         - (16 * np.power(ys_val, 2) * np.power(as_val, 2))
                         - (16 * ac_val * as_val * ys_val * yc_val)
                         - (4 * np.power(as_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(ys_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - (8 * as_val * ys_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((6 * np.power(g2_val, 2))
                             + ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                                * np.power(ys_val, 2))
                               + (2 * np.power(as_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(ys_val, 2))
                                   - (M2_val * as_val * ys_val)))
                         + ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(ys_val, 2))
                                   - (M1_val * as_val * ys_val)))
                         + ((4 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((128 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((808 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((4 / 15) * np.power(g1_val, 2) * sigma1))

        dmD1_sq_dt_2l = (((-8) * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                          * np.power(yd_val, 4))
                         - (4 * (mU1_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ1_sq_val)
                                 + mD1_sq_val) * np.power(yd_val, 2)
                            * np.power(yu_val, 2))
                         - (np.power(yd_val, 2)
                            * (2 * (mD1_sq_val + mQ1_sq_val
                                    + (2 * mHd_sq_val)))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(yd_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (((mL3_sq_val + mE3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mL2_sq_val + mE2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mL1_sq_val + mE1_sq_val)
                                     * np.power(ye_val, 2)))# end trace
                               ))
                         - (16 * np.power(yd_val, 2) * np.power(ad_val, 2))
                         - (16 * au_val * ad_val * yd_val * yu_val)
                         - (4 * np.power(ad_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(yd_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - (8 * ad_val * yd_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((6 * np.power(g2_val, 2))
                             + ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                                * np.power(yd_val, 2))
                               + (2 * np.power(ad_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yd_val, 2))
                                   - (M2_val * ad_val * yd_val)))
                         + ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yd_val, 2))
                                   - (M1_val * ad_val * yd_val)))
                         + ((4 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((128 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((808 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((4 / 15) * np.power(g1_val, 2) * sigma1))

        # Right leptons
        dmE3_sq_dt_2l = (((-8) * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                         * np.power(ytau_val, 4))
            - (np.power(ytau_val, 2)
               * ((2 * mL3_sq_val) + (2 * mE3_sq_val)
                  + (4 * mHd_sq_val))# Tr(6Yd^2 + 2Ye^2)
               * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  + (2 * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                          + np.power(ye_val, 2)))))# end trace
            - (4 * np.power(ytau_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
               * ((3 * (((mQ3_sq_val + mD3_sq_val) * np.power(yb_val, 2))
                        + ((mQ2_sq_val + mD2_sq_val) * np.power(ys_val, 2))
                        + ((mQ1_sq_val + mD1_sq_val) * np.power(yd_val, 2))))
                  + ((((mL3_sq_val + mE3_sq_val) * np.power(ytau_val, 2))
                      + ((mL2_sq_val + mE2_sq_val) * np.power(ymu_val, 2))
                      + ((mL1_sq_val + mE1_sq_val) * np.power(ye_val, 2))))# end trace
                  ))
            - (16 * np.power(ytau_val, 2) * np.power(atau_val, 2))
            - (4 * np.power(atau_val, 2)# Tr(3Yd^2 + Ye^2)
               * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  + ((np.power(ytau_val, 2) + np.power(ymu_val, 2)
                      + np.power(ye_val, 2)))))# end trace
            - (4 * np.power(ytau_val, 2) # Tr(3ad^2 + ae^2)
               * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                        + np.power(ad_val, 2)))
                  + ((np.power(atau_val, 2) + np.power(amu_val, 2)
                      + np.power(ae_val, 2)))))# end trace
            - (8 * atau_val * ytau_val # Tr(3Yd * ad + Ye * ae)
               * ((3 * ((yb_val * ab_val) + (ys_val * as_val)
                         + (yd_val * ad_val)))
                  + (((ytau_val * atau_val) + (ymu_val * amu_val)
                      + (ye_val * ae_val)))))# end trace
            + (((6 * np.power(g2_val, 2)) - (6 / 5) * np.power(g1_val, 2))
               * ((2 * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                   * np.power(ytau_val, 2))
                  + (2 * np.power(atau_val, 2))))
            + (12 * np.power(g2_val, 2) * 2
               * ((np.power(M2_val, 2) * np.power(ytau_val, 2))
                  - (M2_val * atau_val * ytau_val)))
            - ((12 / 5) * np.power(g1_val, 2) * 2
               * ((np.power(M1_val, 2) * np.power(ytau_val, 2))
                  - (M1_val * atau_val * ytau_val)))
            + ((12 / 5) * np.power(g1_val, 2) * Spr_val)
            + ((2808 / 25) * np.power(g1_val, 4) * np.power(M1_val, 2))
            + ((12 / 5) * np.power(g1_val, 2) * sigma1))

        dmE2_sq_dt_2l = (((-8) * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                          * np.power(ymu_val, 4))
                         - (np.power(ymu_val, 2)
                            * ((2 * mL2_sq_val) + (2 * mE2_sq_val)
                               + (4 * mHd_sq_val))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ymu_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + ((((mL3_sq_val + mE3_sq_val)
                                    * np.power(ytau_val, 2))
                                   + ((mL2_sq_val + mE2_sq_val)
                                      * np.power(ymu_val, 2))
                                   + ((mL1_sq_val + mE1_sq_val)
                                      * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ymu_val, 2) * np.power(amu_val, 2))
                         - (4 * np.power(amu_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + ((np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ymu_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + ((np.power(atau_val, 2) + np.power(amu_val, 2)
                                   + np.power(ae_val, 2)))))# end trace
                         - (8 * amu_val * ymu_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (((ytau_val * atau_val) + (ymu_val * amu_val)
                                   + (ye_val * ae_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - (6 / 5) * np.power(g1_val, 2))
                            * ((2 * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                                * np.power(ymu_val, 2))
                               + (2 * np.power(amu_val, 2))))
                         + (12 * np.power(g2_val, 2) * 2
                            * ((np.power(M2_val, 2) * np.power(ymu_val, 2))
                               - (M2_val * amu_val * ymu_val)))
                         - ((12 / 5) * np.power(g1_val, 2) * 2
                            * ((np.power(M1_val, 2) * np.power(ymu_val, 2))
                               - (M1_val * amu_val * ymu_val)))
                         + ((12 / 5) * np.power(g1_val, 2) * Spr_val)
                         + ((2808 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((12 / 5) * np.power(g1_val, 2) * sigma1))

        dmE1_sq_dt_2l = (((-8) * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                          * np.power(ye_val, 4))
                         - (np.power(ye_val, 2)
                            * ((2 * mL1_sq_val) + (2 * mE1_sq_val)
                               + (4 * mHd_sq_val))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ye_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + ((((mL3_sq_val + mE3_sq_val)
                                    * np.power(ytau_val, 2))
                                   + ((mL2_sq_val + mE2_sq_val)
                                      * np.power(ymu_val, 2))
                                   + ((mL1_sq_val + mE1_sq_val)
                                      * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ye_val, 2) * np.power(ae_val, 2))
                         - (4 * np.power(ae_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + ((np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ye_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + ((np.power(atau_val, 2) + np.power(amu_val, 2)
                                   + np.power(ae_val, 2)))))# end trace
                         - (8 * ae_val * ye_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (((ytau_val * atau_val) + (ymu_val * amu_val)
                                   + (ye_val * ae_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - (6 / 5) * np.power(g1_val, 2))
                            * ((2 * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                                * np.power(ye_val, 2))
                               + (2 * np.power(ae_val, 2))))
                         + (12 * np.power(g2_val, 2) * 2
                            * ((np.power(M2_val, 2) * np.power(ye_val, 2))
                               - (M2_val * ae_val * ye_val)))
                         - ((12 / 5) * np.power(g1_val, 2) * 2
                            * ((np.power(M1_val, 2) * np.power(ye_val, 2))
                               - (M1_val * ae_val * ye_val)))
                         + ((12 / 5) * np.power(g1_val, 2) * Spr_val)
                         + ((2808 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((12 / 5) * np.power(g1_val, 2) * sigma1))

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
        # 1 loop part
        # dtanb_dt_1l = 3 * (np.power(yt_val, 2) - np.power(yb_val, 2))

        # 2 loop part
        # dtanb_dt_2l = (((-9) * (np.power(yt_val, 4) - np.power(yb_val, 4)))
        #               + (6 * np.power(yt_val, 2)
        #                  * (((8 / 3) * np.power(g3_val, 2))
        #                     + ((6 / 45) * np.power(g1_val, 2))))
        #               - (6 * np.power(yb_val, 2)
        #                  * (((8 / 3) * np.power(g3_val, 2))
        #                     - ((3 / 45) * np.power(g1_val, 2))))
        #               - (3 * (np.power(yt_val, 2) - np.power(yb_val, 2))
        #                  * (((1 / np.sqrt(2))
        #                      * (((3 / 5) * np.power(g1_val, 2))
        #                         + np.power(g2_val, 2)))
        #                     + np.power(g2_val, 2))))

        # Total beta function for tanb
        # dtanb_dt = (tanb_val / t) * ((loop_fac * dtanb_dt_1l)
        #                             + (loop_fac_sq * dtanb_dt_2l))


        # Collect all for return
        dxdt = [dg1_dt, dg2_dt, dg3_dt, dM1_dt, dM2_dt, dM3_dt, dmu_dt, dyt_dt,
                dyc_dt, dyu_dt, dyb_dt, dys_dt, dyd_dt, dytau_dt, dymu_dt,
                dye_dt, dat_dt, dac_dt, dau_dt, dab_dt, das_dt, dad_dt,
                datau_dt, damu_dt, dae_dt, dmHu_sq_dt, dmHd_sq_dt,
                dmQ1_sq_dt, dmQ2_sq_dt, dmQ3_sq_dt, dmL1_sq_dt, dmL2_sq_dt,
                dmL3_sq_dt, dmU1_sq_dt, dmU2_sq_dt, dmU3_sq_dt, dmD1_sq_dt,
                dmD2_sq_dt, dmD3_sq_dt, dmE1_sq_dt, dmE2_sq_dt, dmE3_sq_dt]#,
                #dtanb_dt]
        return dxdt

    # Set up domains for solve_ivp
    t_span = np.array([my_QGUT*1.0000001, weak_scale * 0.999999])

    # Now solve
    sol = solve_ivp(my_odes, t_span, GUT_BCs, t_eval = t_vals[::-1],
                    dense_output=True, method='DOP853', atol=1e-9, rtol=1e-9)

    #t = sol.t
    x = sol.y
    return x

def my_RGE_solver_2(weak_BCs, my_QGUT, low_Q_val):
    """
    Use scipy.integrate to evolve MSSM RGEs and collect solution vectors.

    Parameters
    ----------
    weak_BCs : Array of floats.
        weak scale boundary conditions for RGEs.
    my_QGUT : Float.
        Highest value for t parameter to run to in solution,
            typically unification scale from SoftSUSY.
    low_Q_val : Float.
        Lowest value for t parameter to run to in solution. This is
        where BC's are defined in this second run.

    Returns
    -------
    Array of floats.
        Return solutions to system of RGEs.

    """
    def my_odes_2(t, x):
        """
        Define two-loop RGEs for soft terms.

        Parameters
        ----------
        x : Array of floats.
            Numerical solutions to RGEs. The order of entries in x is:
              (0: g1, 1: g2, 2: g3, 3: M1, 4: M2, 5: M3, 6: mu, 7: yt, 8: yc,
               9: yu, 10: yb, 11: ys, 12: yd, 13: ytau, 14: ymu, 15: ye,
               16: at, 17: ac, 18: au, 19: ab, 20: as, 21: ad, 22: atau,
               23: amu, 24: ae, 25: mHu^2, 26: mHd^2, 27: mQ1^2,
               28: mQ2^2, 29: mQ3^2, 30: mL1^2, 31: mL2^2, 32: mL3^2,
               33: mU1^2, 34: mU2^2, 35: mU3^2, 36: mD1^2, 37: mD2^2,
               38: mD3^2, 39: mE1^2, 40: mE2^2, 41: mE3^2, 42: b)
        t : Array of evaluation renormalization scales.
            t = Q values for numerical solutions.

        Returns
        -------
        Array of floats.
            Return all soft RGEs evaluated at current t value.

        """
        # Unification scale is acquired from running a BM point through
        # SoftSUSY, then GUT scale boundary conditions are acquired from
        # SoftSUSY so that all three generations of Yukawas (assumed
        # to be diagonalized) are accounted for. A universal boundary condition
        # is used for soft scalar trilinear couplings a_i=y_i*A_i.
        # The soft b^(ij) mass^2 term is defined as b=B*mu, but is computed
        # in a later iteration.
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
        mHu_sq_val = x[25]
        mHd_sq_val = x[26]
        mQ1_sq_val = x[27]
        mQ2_sq_val = x[28]
        mQ3_sq_val = x[29]
        mL1_sq_val = x[30]
        mL2_sq_val = x[31]
        mL3_sq_val = x[32]
        mU1_sq_val = x[33]
        mU2_sq_val = x[34]
        mU3_sq_val = x[35]
        mD1_sq_val = x[36]
        mD2_sq_val = x[37]
        mD3_sq_val = x[38]
        mE1_sq_val = x[39]
        mE2_sq_val = x[40]
        mE3_sq_val = x[41]
        b_val = x[42]

        ##### Gauge couplings and gaugino masses #####
        # 1 loop parts
        dg1_dt_1l = b_1l[0] * np.power(g1_val, 3)

        dg2_dt_1l = b_1l[1] * np.power(g2_val, 3)

        dg3_dt_1l = b_1l[2] * np.power(g3_val, 3)

        dM1_dt_1l = b_1l[0] * np.power(g1_val, 2) * M1_val

        dM2_dt_1l = b_1l[1] * np.power(g2_val, 2) * M2_val

        dM3_dt_1l = b_1l[2] * np.power(g3_val, 2) * M3_val

        # 2 loop parts
        dg1_dt_2l = (np.power(g1_val, 3)
                     * ((b_2l[0][0] * np.power(g1_val, 2))
                        + (b_2l[0][1] * np.power(g2_val, 2))
                        + (b_2l[0][2] * np.power(g3_val, 2))# Tr(Yu^2)
                        - (c_2l[0][0] * (np.power(yt_val, 2)
                                         + np.power(yc_val, 2)
                                         + np.power(yu_val, 2)))# end trace, begin Tr(Yd^2)
                        - (c_2l[0][1] * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))# end trace, begin Tr(Ye^2)
                        - (c_2l[0][2] * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace

        dg2_dt_2l = (np.power(g2_val, 3)
                     * ((b_2l[1][0] * np.power(g1_val, 2))
                        + (b_2l[1][1] * np.power(g2_val, 2))
                        + (b_2l[1][2] * np.power(g3_val, 2))# Tr(Yu^2)
                        - (c_2l[1][0] * (np.power(yt_val, 2)
                                         + np.power(yc_val, 2)
                                         + np.power(yu_val, 2)))# end trace, begin Tr(Yd^2)
                        - (c_2l[1][1] * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))# end trace, begin Tr(Ye^2)
                        - (c_2l[1][2] * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace

        dg3_dt_2l = (np.power(g3_val, 3)
                     * ((b_2l[2][0] * np.power(g1_val, 2))
                        + (b_2l[2][1] * np.power(g2_val, 2))
                        + (b_2l[2][2] * np.power(g3_val, 2))# Tr(Yu^2)
                        - (c_2l[2][0] * (np.power(yt_val, 2)
                                         + np.power(yc_val, 2)
                                         + np.power(yu_val, 2)))# end trace, begin Tr(Yd^2)
                        - (c_2l[2][1] * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))# end trace, begin Tr(Ye^2)
                        - (c_2l[2][2] * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace

        dM1_dt_2l = (2 * np.power(g1_val, 2)
                     * (((b_2l[0][0] * np.power(g1_val, 2) * (M1_val + M1_val))
                         + (b_2l[0][1] * np.power(g2_val, 2)
                            * (M1_val + M2_val))
                         + (b_2l[0][2] * np.power(g3_val, 2)
                            * (M1_val + M3_val)))# Tr(Yu*au)
                        + ((c_2l[0][0] * (((yt_val * at_val)
                                           + (yc_val * ac_val)
                                           + (yu_val * au_val))# end trace, begin Tr(Yu^2)
                                          - (M1_val * (np.power(yt_val, 2)
                                                       + np.power(yc_val, 2)
                                                       + np.power(yu_val, 2)))# end trace
                                          )))# Tr(Yd*ad)
                        + ((c_2l[0][1] * (((yb_val * ab_val)
                                           + (ys_val * as_val)
                                           + (yd_val * ad_val))# end trace, begin Tr(Yd^2)
                                          - (M1_val * (np.power(yb_val, 2)
                                                       + np.power(ys_val, 2)
                                                       + np.power(yd_val, 2)))# end trace
                                          )))# Tr(Ye*ae)
                        + ((c_2l[0][2] * (((ytau_val * atau_val)
                                           + (ymu_val * amu_val)
                                           + (ye_val * ae_val))# end trace, begin Tr(Ye^2)
                                          - (M1_val * (np.power(ytau_val, 2)
                                                       + np.power(ymu_val, 2)
                                                       + np.power(ye_val, 2)))
                                          )))))

        dM2_dt_2l = (2 * np.power(g2_val, 2)
                     * (((b_2l[1][0] * np.power(g1_val, 2) * (M2_val + M1_val))
                         + (b_2l[1][1] * np.power(g2_val, 2)
                            * (M2_val + M2_val))
                         + (b_2l[1][2] * np.power(g3_val, 2)
                            * (M2_val + M3_val)))# Tr(Yu*au)
                        + ((c_2l[1][0] * (((yt_val * at_val)
                                           + (yc_val * ac_val)
                                           + (yu_val * au_val))# end trace, begin Tr(Yu^2)
                                          - (M2_val * (np.power(yt_val, 2)
                                                       + np.power(yc_val, 2)
                                                       + np.power(yu_val, 2)))# end trace
                                          )))# Tr(Yd*ad)
                        + ((c_2l[1][1] * (((yb_val * ab_val)
                                           + (ys_val * as_val)
                                           + (yd_val * ad_val))# end trace, begin Tr(Yd^2)
                                          - (M2_val * (np.power(yb_val, 2)
                                                       + np.power(ys_val, 2)
                                                       + np.power(yd_val, 2)))# end trace
                                          )))# Tr(Ye*ae)
                        + ((c_2l[1][2] * (((ytau_val * atau_val)
                                           + (ymu_val * amu_val)
                                           + (ye_val * ae_val))# end trace, begin Tr(Ye^2)
                                          - (M2_val * (np.power(ytau_val, 2)
                                                       + np.power(ymu_val, 2)
                                                       + np.power(ye_val, 2)))# end trace
                                          )))))

        dM3_dt_2l = (2 * np.power(g3_val, 2)
                     * (((b_2l[2][0] * np.power(g1_val, 2) * (M3_val + M1_val))
                         + (b_2l[2][1] * np.power(g2_val, 2)
                            * (M3_val + M2_val))
                         + (b_2l[2][2] * np.power(g3_val, 2)
                            * (M3_val + M3_val)))# Tr(Yu*au)
                        + ((c_2l[2][0] * (((yt_val * at_val)
                                           + (yc_val * ac_val)
                                           + (yu_val * au_val))# end trace, begin Tr(Yu^2)
                                          - (M3_val * (np.power(yt_val, 2)
                                                       + np.power(yc_val, 2)
                                                       + np.power(yu_val, 2)))# end trace
                                          )))# Tr(Yd*ad)
                        + ((c_2l[2][1] * (((yb_val * ab_val)
                                           + (ys_val * as_val)
                                           + (yd_val * ad_val))# end trace, begin Tr(Yd^2)
                                          - (M3_val * (np.power(yb_val, 2)
                                                       + np.power(ys_val, 2)
                                                       + np.power(yd_val, 2)))# end trace
                                          )))# Tr(Ye*ae)
                        + ((c_2l[2][2] * (((ytau_val * atau_val)
                                           + (ymu_val * amu_val)
                                           + (ye_val * ae_val))# end trace, begin Tr(Ye^2)
                                          - (M3_val * (np.power(ytau_val, 2)
                                                       + np.power(ymu_val, 2)
                                                       + np.power(ye_val, 2)))# end trace
                                          )))))

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
        dmu_dt_1l = (mu_val# Tr(3Yu^2 + 3Yd^2 + Ye^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2) + np.power(yb_val, 2)
                              + np.power(ys_val, 2) + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        - (3 * np.power(g2_val, 2))
                        - ((3 / 5) * np.power(g1_val, 2))))

        # 2 loop part
        dmu_dt_2l = (mu_val# Tr(3Yu^4 + 3Yd^4 + (2Yu^2*Yd^2) + Ye^4)
                     * ((-3 * ((3 * (np.power(yt_val, 4) + np.power(yc_val, 4)
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
                               + (np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4))))# end trace
                        + (((16 * np.power(g3_val, 2))
                            + (4 * np.power(g1_val, 2) / 5))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((16 * np.power(g3_val, 2))
                            - (2 * np.power(g1_val, 2) / 5))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((15 / 2) * np.power(g2_val, 4))
                        + ((9 / 5) * np.power(g1_val, 2)
                           * np.power(g2_val, 2))
                        + ((207 / 50) * np.power(g1_val, 4))))

        # Total mu beta function
        dmu_dt = (1 / t) * ((loop_fac * dmu_dt_1l)
                            + (loop_fac_sq * dmu_dt_2l))

        ##### Yukawa couplings for all 3 generations, assumed diagonalized#####
        # 1 loop parts
        dyt_dt_1l = (yt_val# Tr(3Yu^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (3 * (np.power(yt_val, 2)))
                        + np.power(yb_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))

        dyc_dt_1l = (yc_val# Tr(3Yu^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (3 * (np.power(yc_val, 2)))
                        + np.power(ys_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))

        dyu_dt_1l = (yu_val# Tr(3Yu^2)
                     * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (3 * (np.power(yu_val, 2)))
                        + np.power(yd_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((13 / 15) * np.power(g1_val, 2))))

        dyb_dt_1l = (yb_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(yb_val, 2))) + np.power(yt_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))

        dys_dt_1l = (ys_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(ys_val, 2))) + np.power(yc_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))

        dyd_dt_1l = (yd_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(yd_val, 2))) + np.power(yu_val, 2)
                        - ((16 / 3) * np.power(g3_val, 2))
                        - (3 * np.power(g2_val, 2))
                        - ((7 / 15) * np.power(g1_val, 2))))

        dytau_dt_1l = (ytau_val# Tr(3Yd^2 + Ye^2)
                       * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2))# end trace
                          + (3 * (np.power(ytau_val, 2)))
                          - (3 * np.power(g2_val, 2))
                          - ((9 / 5) * np.power(g1_val, 2))))

        dymu_dt_1l = (ymu_val# Tr(3Yd^2 + Ye^2)
                      * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))
                         + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                            + np.power(ye_val, 2))# end trace
                         + (3 * (np.power(ymu_val, 2)))
                         - (3 * np.power(g2_val, 2))
                         - ((9 / 5) * np.power(g1_val, 2))))

        dye_dt_1l = (ye_val# Tr(3Yd^2 + Ye^2)
                     * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))
                        + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                           + np.power(ye_val, 2))# end trace
                        + (3 * (np.power(ye_val, 2)))
                        - (3 * np.power(g2_val, 2))
                        - ((9 / 5) * np.power(g1_val, 2))))

        # 2 loop parts
        dyt_dt_2l = (yt_val # Tr(3Yu^4 + (Yu^2*Yd^2))
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2)))
                         )# end trace
                        - (np.power(yb_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))# end trace
                        - (9 * np.power(yt_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (4 * np.power(yt_val, 4))
                        - (2 * np.power(yb_val, 4))
                        - (2 * np.power(yb_val, 2) * np.power(yt_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((6 * np.power(g2_val, 2))
                            + ((2 / 5) * np.power(g1_val, 2)))
                           * np.power(yt_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2) * np.power(yb_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))

        dyc_dt_2l = (yc_val # Tr(3Yu^4 + (Yu^2*Yd^2))
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2)))
                         )#end trace
                        - (np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))# end trace
                        - (9 * np.power(yc_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (4 * np.power(yc_val, 4))
                        - (2 * np.power(ys_val, 4))
                        - (2 * np.power(ys_val, 2)
                           * np.power(yc_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((6 * np.power(g2_val, 2))
                            + ((2 / 5) * np.power(g1_val, 2)))
                           * np.power(yc_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2) * np.power(ys_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))

        dyu_dt_2l = (yu_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                     * (((-3) * ((3 * (np.power(yt_val, 4)
                                       + np.power(yc_val, 4)
                                       + np.power(yu_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2)))
                         )# end trace
                        - (np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + (np.power(ytau_val, 2)
                                 + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2))))
                        - (9 * np.power(yu_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))
                        - (4 * np.power(yu_val, 4))
                        - (2 * np.power(yd_val, 4))
                        - (2 * np.power(yd_val, 2) * np.power(yu_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((6 * np.power(g2_val, 2))
                            + ((2 / 5) * np.power(g1_val, 2)))
                           * np.power(yu_val, 2))
                        + ((2 / 5) * np.power(g1_val, 2) * np.power(yd_val, 2))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((136 / 45) * np.power(g3_val, 2)
                           * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((2743 / 450) * np.power(g1_val, 4))))

        dyb_dt_2l = (yb_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(yt_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (3 * np.power(yb_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(yb_val, 4))
                        - (2 * np.power(yt_val, 4))
                        - (2 * np.power(yt_val, 2) * np.power(yb_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((4 / 5) * np.power(g1_val, 2)
                           * np.power(yt_val, 2))
                        + (np.power(yb_val, 2)
                           * ((6 * np.power(g2_val, 2))
                              + ((4 / 5) * np.power(g1_val, 2))))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))

        dys_dt_2l = (ys_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4)
                                 + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(yc_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (3 * np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(ys_val, 4))
                        - (2 * np.power(yc_val, 4))
                        - (2 * np.power(yc_val, 2) * np.power(ys_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((4 / 5) * np.power(g1_val, 2) * np.power(yc_val, 2))
                        + (np.power(ys_val, 2)
                           * ((6 * np.power(g2_val, 2))
                              + ((4 / 5) * np.power(g1_val, 2))))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))

        dyd_dt_2l = (yd_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2) * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2) * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2) * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4) + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(yu_val, 2)# Tr(Yu^2)
                           * (np.power(yt_val, 2)
                              + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        - (3 * np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(yd_val, 4))
                        - (2 * np.power(yu_val, 4))
                        - (2 * np.power(yd_val, 2) * np.power(yu_val, 2))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((4 / 5) * np.power(g1_val, 2) * np.power(yu_val, 2))
                        + (np.power(yd_val, 2)
                           * ((6 * np.power(g2_val, 2))
                              + ((4 / 5) * np.power(g1_val, 2))))
                        - ((16 / 9) * np.power(g3_val, 4))
                        + (8 * np.power(g3_val, 2) * np.power(g2_val, 2))
                        + ((8 / 9) * np.power(g3_val, 2) * np.power(g1_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + (np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((287 / 90) * np.power(g1_val, 4))))

        dytau_dt_2l = (ytau_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                       * (((-3) * ((3 * (np.power(yb_val, 4)
                                         + np.power(ys_val, 4)
                                         + np.power(yd_val, 4)))
                                   + (np.power(yt_val, 2)
                                      * np.power(yb_val, 2))
                                   + (np.power(yc_val, 2)
                                      * np.power(ys_val, 2))
                                   + (np.power(yu_val, 2)
                                      * np.power(yd_val, 2))
                                   + np.power(ytau_val, 4)
                                   + np.power(ymu_val, 4)
                                   + np.power(ye_val, 4)))# end trace
                          - (3 * np.power(ytau_val, 2)# Tr(3Yd^2 + Ye^2)
                             * ((3 * (np.power(yb_val, 2)
                                      + np.power(ys_val, 2)
                                      + np.power(yd_val, 2)))
                                + np.power(ytau_val, 2)
                                + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))# end trace
                          - (4 * np.power(ytau_val, 4))
                          + (((16 * np.power(g3_val, 2))
                              - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                             * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))# end trace
                          + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                             * (np.power(ytau_val, 2)
                                + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))# end trace
                          + (6 * np.power(g2_val, 2) * np.power(ytau_val, 2))
                          + ((15 / 2) * np.power(g2_val, 4))
                          + ((9 / 5) * np.power(g2_val, 2)
                             * np.power(g1_val, 2))
                          + ((27 / 2) * np.power(g1_val, 4))))

        dymu_dt_2l = (ymu_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                      * (((-3) * ((3 * (np.power(yb_val, 4)
                                        + np.power(ys_val, 4)
                                        + np.power(yd_val, 4)))
                                  + (np.power(yt_val, 2)
                                     * np.power(yb_val, 2))
                                  + (np.power(yc_val, 2)
                                     * np.power(ys_val, 2))
                                  + (np.power(yu_val, 2)
                                     * np.power(yd_val, 2))
                                  + np.power(ytau_val, 4)
                                  + np.power(ymu_val, 4)
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(ymu_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2)
                               + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(ymu_val, 4))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2)
                               + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (6 * np.power(g2_val, 2) * np.power(ymu_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + ((9 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2))
                         + ((27 / 2) * np.power(g1_val, 4))))

        dye_dt_2l = (ye_val# Tr(3Yd^4 + (Yu^2*Yd^2) + Ye^4)
                     * (((-3) * ((3 * (np.power(yb_val, 4)
                                       + np.power(ys_val, 4)
                                       + np.power(yd_val, 4)))
                                 + (np.power(yt_val, 2)
                                    * np.power(yb_val, 2))
                                 + (np.power(yc_val, 2)
                                    * np.power(ys_val, 2))
                                 + (np.power(yu_val, 2)
                                    * np.power(yd_val, 2))
                                 + np.power(ytau_val, 4)
                                 + np.power(ymu_val, 4)
                                 + np.power(ye_val, 4)))# end trace
                        - (3 * np.power(ye_val, 2)# Tr(3Yd^2 + Ye^2)
                           * ((3 * (np.power(yb_val, 2)
                                    + np.power(ys_val, 2)
                                    + np.power(yd_val, 2)))
                              + np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        - (4 * np.power(ye_val, 4))
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2)
                              + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))
                        + (6 * np.power(g2_val, 2) * np.power(ye_val, 2))
                        + ((15 / 2) * np.power(g2_val, 4))
                        + ((9 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2))
                        + ((27 / 2) * np.power(g1_val, 4))))

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
        dat_dt_1l = ((at_val# Tr(Yu^2)
                      * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (5 * np.power(yt_val, 2)) + np.power(yb_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((13 / 15) * np.power(g1_val, 2))))
                     + (yt_val# Tr(au*Yu)
                        * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (4 * yt_val * at_val)
                           + (2 * yb_val * ab_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((26 / 15) * np.power(g1_val, 2) * M1_val))))

        dac_dt_1l = ((ac_val# Tr(Yu^2)
                      * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (5 * np.power(yc_val, 2)) + np.power(ys_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((13 / 15) * np.power(g1_val, 2))))
                     + (yc_val# Tr(au*Yu)
                        * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (4 * yc_val * ac_val)
                           + (2 * ys_val * as_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((26 / 15) * np.power(g1_val, 2) * M1_val))))

        dau_dt_1l = ((au_val# Tr(Yu^2)
                      * ((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         + (5 * np.power(yu_val, 2)) + np.power(yd_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((13 / 15) * np.power(g1_val, 2))))
                     + (yu_val# Tr(au*Yu)
                        * ((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (4 * yu_val * au_val)
                           + (2 * yd_val * ad_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((26 / 15) * np.power(g1_val, 2) * M1_val))))

        dab_dt_1l = ((ab_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(yb_val, 2)) + np.power(yt_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((7 / 15) * np.power(g1_val, 2))))
                     + (yb_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                           + (4 * yb_val * ab_val) + (2 * yt_val * at_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((14 / 15) * np.power(g1_val, 2) * M1_val))))

        das_dt_1l = ((as_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(ys_val, 2)) + np.power(yc_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((7 / 15) * np.power(g1_val, 2))))
                     + (ys_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))
                           + (4 * ys_val * as_val) + (2 * yc_val * ac_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((14 / 15) * np.power(g1_val, 2) * M1_val))))

        dad_dt_1l = ((ad_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(yd_val, 2)) + np.power(yu_val, 2)
                         - ((16 / 3) * np.power(g3_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((7 / 15) * np.power(g1_val, 2))))
                     + (yd_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                           + (4 * yd_val * ad_val) + (2 * yu_val * au_val)
                           + ((32 / 3) * np.power(g3_val, 2) * M3_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((14 / 15) * np.power(g1_val, 2) * M1_val))))

        datau_dt_1l = ((atau_val# Tr(3Yd^2 + Ye^2)
                        * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                  + np.power(yd_val, 2)))
                            + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                           + (5 * np.power(ytau_val, 2))
                           - (3 * np.power(g2_val, 2))
                           - ((9 / 5) * np.power(g1_val, 2))))
                       + (ytau_val# Tr(6ad*Yd + 2ae*Ye)
                          * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                   + (ad_val * yd_val)))
                             + (2 * ((atau_val * ytau_val)
                                     + (amu_val * ymu_val)
                                     + (ae_val * ye_val)))# end trace
                             + (4 * ytau_val * atau_val)
                             + (6 * np.power(g2_val, 2) * M2_val)
                             + ((18 / 5) * np.power(g1_val, 2) * M1_val))))

        damu_dt_1l = ((amu_val# Tr(3Yd^2 + Ye^2)
                       * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))
                           + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                          + (5 * np.power(ymu_val, 2))
                          - (3 * np.power(g2_val, 2))
                          - ((9 / 5) * np.power(g1_val, 2))))
                      + (ymu_val# Tr(6ad*Yd + 2ae*Ye)
                         * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                  + (ad_val * yd_val)))
                            + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val)))# end trace
                            + (4 * ymu_val * amu_val)
                            + (6 * np.power(g2_val, 2) * M2_val)
                            + ((18 / 5) * np.power(g1_val, 2) * M1_val))))

        dae_dt_1l = ((ae_val# Tr(3Yd^2 + Ye^2)
                      * (((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))
                          + (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                             + np.power(ye_val, 2)))# end trace
                         + (5 * np.power(ye_val, 2))
                         - (3 * np.power(g2_val, 2))
                         - ((9 / 5) * np.power(g1_val, 2))))
                     + (ye_val# Tr(6ad*Yd + 2ae*Ye)
                        * ((6 * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                           + (4 * ye_val * ae_val)
                           + (6 * np.power(g2_val, 2) * M2_val)
                           + ((18 / 5) * np.power(g1_val, 2) * M1_val))))

        # 2 loop parts
        dat_dt_2l = ((at_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                      * (((-3) * ((3 * (np.power(yt_val, 4)
                                        + np.power(yc_val, 4)
                                        + np.power(yu_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))))# end trace
                         - (np.power(yb_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (15 * np.power(yt_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (6 * np.power(yt_val, 4))
                         - (2 * np.power(yb_val, 4))
                         - (4 * np.power(yb_val, 2) * np.power(yt_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
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
                         + ((2743 / 450) * np.power(g1_val, 4))))
                     + (yt_val# Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                        * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                          + (ac_val * np.power(yc_val, 3))
                                          + (au_val * np.power(yu_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)))# end trace
                           - (18 * np.power(yt_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (np.power(yb_val, 2)# Tr(6ad*Yd + 2ae*Ye)
                              * ((6
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + (2
                                    * ((atau_val * ytau_val)
                                       + (amu_val * ymu_val)
                                       + (ae_val * ye_val)))))# end trace
                           - (12 * yt_val * at_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (yb_val * ab_val# Tr(6Yd^2 + 2Ye^2)
                              * ((6 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (2 * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yt_val, 3) * at_val)
                           - (8 * np.power(yb_val, 3) * ab_val)
                           - (2 * np.power(yb_val, 2) * yt_val * at_val)
                           - (4 * yb_val * ab_val * np.power(yt_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                              * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yt_val * at_val)
                           + ((4 / 5) * np.power(g1_val, 2) * yb_val * ab_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yt_val, 2))
                           - ((4 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yb_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((272 / 45) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((5486 / 225) * np.power(g1_val, 4) * M1_val))))

        dac_dt_2l = ((ac_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                      * (((-3) * ((3 * (np.power(yt_val, 4)
                                        + np.power(yc_val, 4)
                                        + np.power(yu_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))))# end trace
                         - (np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (15 * np.power(yc_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (6 * np.power(yc_val, 4))
                         - (2 * np.power(ys_val, 4))
                         - (4 * np.power(ys_val, 2) * np.power(yc_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
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
                         + ((2743 / 450) * np.power(g1_val, 4))))
                     + (yc_val# Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                        * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                          + (ac_val * np.power(yc_val, 3))
                                          + (au_val * np.power(yu_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)))# end trace
                           - (18 * np.power(yc_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (np.power(ys_val, 2)# Tr(6ad*Yd + 2ae*Ye)
                              * ((6
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + (2
                                    * ((atau_val * ytau_val)
                                       + (amu_val * ymu_val)
                                       + (ae_val * ye_val)))))# end trace
                           - (12 * yc_val * ac_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (ys_val * as_val# Tr(6Yd^2 + 2Ye^2)
                              * ((6 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (2 * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yc_val, 3) * ac_val)
                           - (8 * np.power(ys_val, 3) * as_val)
                           - (2 * np.power(ys_val, 2) * yc_val * ac_val)
                           - (4 * ys_val * as_val * np.power(yc_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                              * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yc_val * ac_val)
                           + ((4 / 5) * np.power(g1_val, 2) * ys_val * as_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yc_val, 2))
                           - ((4 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(ys_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((272 / 45) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((5486 / 225) * np.power(g1_val, 4) * M1_val))))

        dau_dt_2l = ((au_val# Tr(3Yu^4 + (Yu^2*Yd^2))
                      * (((-3) * ((3 * (np.power(yt_val, 4)
                                        + np.power(yc_val, 4)
                                        + np.power(yu_val, 4)))
                                  + ((np.power(yt_val, 2) * np.power(yb_val,2))
                                     + (np.power(yc_val, 2)
                                        * np.power(ys_val, 2))
                                     + (np.power(yu_val, 2)
                                        * np.power(yd_val, 2)))))# end trace
                         - (np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (15 * np.power(yu_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (6 * np.power(yu_val, 4))
                         - (2 * np.power(yd_val, 4))
                         - (4 * np.power(yd_val, 2) * np.power(yu_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                            * (np.power(yt_val, 2)
                               + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
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
                         + ((2743 / 450) * np.power(g1_val, 4))))
                     + (yu_val# Tr(6au*Yu^3 + au*Yd^2*Yu + ad*Yu^2*Yd)
                        * (((-6) * ((6 * ((at_val * np.power(yt_val, 3))
                                          + (ac_val * np.power(yc_val, 3))
                                          + (au_val * np.power(yu_val, 3))))
                                    + (at_val * np.power(yb_val, 2) * yt_val)
                                    + (ac_val * np.power(ys_val, 2) * yc_val)
                                    + (au_val * np.power(yd_val, 2) * yu_val)
                                    + (ab_val * np.power(yt_val, 2) * yb_val)
                                    + (as_val * np.power(yc_val, 2) * ys_val)
                                    + (ad_val * np.power(yu_val, 2) * yd_val)))# end trace
                           - (18 * np.power(yu_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (np.power(yd_val, 2)# Tr(6ad*Yd + 2ae*Ye)
                              * ((6
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + (2
                                    * ((atau_val * ytau_val)
                                       + (amu_val * ymu_val)
                                       + (ae_val * ye_val)))))# end trace
                           - (12 * yu_val * au_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (yd_val * ad_val# Tr(6Yd^2 + 2Ye^2)
                              * ((6 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (2 * (np.power(ytau_val, 2)
                                         + np.power(ymu_val, 2)
                                         + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yu_val, 3) * au_val)
                           - (8 * np.power(yd_val, 3) * ad_val)
                           - (2 * np.power(yd_val, 2) * yu_val * au_val)
                           - (4 * yd_val * ad_val * np.power(yu_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                              * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yu_val * au_val)
                           + ((4 / 5) * np.power(g1_val, 2) * yd_val * ad_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((4 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yu_val, 2))
                           - ((4 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yd_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((272 / 45) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((5486 / 225) * np.power(g1_val, 4) * M1_val))))

        dab_dt_2l = ((ab_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
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
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(yt_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (5 * np.power(yb_val, 2)# Tr(3Yd^2+Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(yb_val, 4))
                         - (2 * np.power(yt_val, 4))
                         - (4 * np.power(yb_val, 2) * np.power(yt_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
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
                         + ((287 / 90) * np.power(g1_val, 4))))
                     + (yb_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
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
                                            + (ae_val * np.power(ye_val, 3)))
                                       )))# end trace
                           - (6 * np.power(yt_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (6 * np.power(yb_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))# end trace
                           - (6 * yt_val * at_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (4 * yb_val * ab_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yb_val, 3) * ab_val)
                           - (8 * np.power(yt_val, 3) * at_val)
                           - (4 * np.power(yb_val, 2) * yt_val * at_val)
                           - (2 * yb_val * ab_val * np.power(yt_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + ((8 / 5) * np.power(g1_val, 2) * yt_val * at_val)
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yb_val * ab_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yb_val, 2))
                           - ((8 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yt_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((16 / 9) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((574 / 45) * np.power(g1_val, 4) * M1_val))))

        das_dt_2l = ((as_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
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
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(yc_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (5 * np.power(ys_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(ys_val, 4))
                         - (2 * np.power(yc_val, 4))
                         - (4 * np.power(ys_val, 2) * np.power(yc_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
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
                         + ((287 / 90) * np.power(g1_val, 4))))
                     + (ys_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
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
                                            + (ae_val * np.power(ye_val, 3)))
                                       )))# end trace
                           - (6 * np.power(yc_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (6 * np.power(ys_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))# end trace
                           - (6 * yc_val * ac_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (4 * ys_val * as_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(ys_val, 3) * as_val)
                           - (8 * np.power(yc_val, 3) * ac_val)
                           - (4 * np.power(ys_val, 2) * yc_val * ac_val)
                           - (2 * ys_val * as_val * np.power(yc_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + ((8 / 5) * np.power(g1_val, 2) * yc_val * ac_val)
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * ys_val * as_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(ys_val, 2))
                           - ((8 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yc_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((16 / 9) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((574 / 45) * np.power(g1_val, 4) * M1_val))))

        dad_dt_2l = ((ad_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
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
                                  + np.power(ye_val, 4)))# end trace
                         - (3 * np.power(yu_val, 2)# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))# end trace
                         - (5 * np.power(yd_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(yd_val, 4))
                         - (2 * np.power(yu_val, 4))
                         - (4 * np.power(yd_val, 2) * np.power(yu_val, 2))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
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
                         + ((287 / 90) * np.power(g1_val, 4))))
                     + (yd_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
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
                                            + (ae_val * np.power(ye_val, 3)))
                                       )))# end trace
                           - (6 * np.power(yu_val, 2)# Tr(au*Yu)
                              * ((at_val * yt_val)
                                 + (ac_val * yc_val)
                                 + (au_val * yu_val)))# end trace
                           - (6 * np.power(yd_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3
                                  * ((ab_val * yb_val) + (as_val * ys_val)
                                     + (ad_val * yd_val)))
                                 + ((atau_val * ytau_val) + (amu_val * ymu_val)
                                    + (ae_val * ye_val))))# end trace
                           - (6 * yu_val * au_val# Tr(Yu^2)
                              * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                 + np.power(yu_val, 2)))# end trace
                           - (4 * yd_val * ad_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (14 * np.power(yd_val, 3) * ad_val)
                           - (8 * np.power(yu_val, 3) * au_val)
                           - (4 * np.power(yd_val, 2) * yu_val * au_val)
                           - (2 * yd_val * ad_val * np.power(yu_val, 2))
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + ((8 / 5) * np.power(g1_val, 2) * yu_val * au_val)
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * yd_val * ad_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (((12 * np.power(g2_val, 2) * M2_val)
                               + ((8 / 5) * np.power(g1_val, 2) * M1_val))
                              * np.power(yd_val, 2))
                           - ((8 / 5) * np.power(g1_val, 2) * M1_val
                              * np.power(yu_val, 2))
                           + ((64 / 9) * np.power(g3_val, 4) * M3_val)
                           - (16 * np.power(g3_val, 2) * np.power(g2_val, 2)
                              * (M3_val + M2_val))
                           - ((16 / 9) * np.power(g3_val, 2)
                              * np.power(g1_val, 2)
                              * (M3_val + M1_val))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - (2 * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M2_val + M1_val))
                           - ((574 / 45) * np.power(g1_val, 4) * M1_val))))

        datau_dt_2l = ((atau_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
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
                                    + np.power(ye_val, 4)))# end trace
                           - (5 * np.power(ytau_val, 2)# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + (np.power(ytau_val, 2)
                                    + np.power(ymu_val, 2)
                                    + np.power(ye_val, 2))))# end trace
                           - (6 * np.power(ytau_val, 4))
                           + (((16 * np.power(g3_val, 2))
                               - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                              * (np.power(yb_val, 2)
                                 + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           + (((12 * np.power(g2_val, 2))
                               - ((6 / 5) * np.power(g1_val, 2)))
                              * np.power(ytau_val, 2))
                           + ((15 / 2) * np.power(g2_val, 4))
                           + ((9 / 5) * np.power(g2_val, 2)
                              * np.power(g1_val, 2))
                           + ((27 / 2) * np.power(g1_val, 4))))
                       + (ytau_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
                          * (((-6) * ((6 * ((ab_val * np.power(yb_val, 3))
                                            + (as_val * np.power(ys_val, 3))
                                            + (ad_val * np.power(yd_val, 3))))
                                      + (at_val * np.power(yb_val, 2) * yt_val)
                                      + (ac_val * np.power(ys_val, 2) * yc_val)
                                      + (au_val * np.power(yd_val, 2) * yu_val)
                                      + (ab_val * np.power(yt_val, 2) * yb_val)
                                      + (as_val * np.power(yc_val, 2) * ys_val)
                                      + (ad_val * np.power(yu_val, 2) * yd_val)
                                      + (2 * ((atau_val
                                               * np.power(ytau_val, 3))
                                              + (amu_val
                                                 * np.power(ymu_val, 3))
                                              + (ae_val
                                                 * np.power(ye_val, 3)))
                                         )))# end trace
                             - (4 * ytau_val * atau_val# Tr(3Yd^2 + Ye^2)
                                * ((3 * (np.power(yb_val, 2)
                                         + np.power(ys_val, 2)
                                         + np.power(yd_val, 2)))
                                   + ((np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                             - (6 * np.power(ytau_val, 2)# Tr(3ad*Yd + ae*Ye)
                                * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                                         + (ad_val * yd_val)))
                                   + (atau_val * ytau_val)
                                   + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                             - (14 * np.power(ytau_val, 3) * atau_val)
                             + (((32 * np.power(g3_val, 2))
                                 - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                                * ((ab_val * yb_val) + (as_val * ys_val)
                                   + (ad_val * yd_val)))# end trace
                             + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                                * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val)))# end trace
                             + (((6 * np.power(g2_val, 2))
                                 + ((6 / 5) * np.power(g1_val, 2)))
                                * ytau_val * atau_val)
                             - (((32 * np.power(g3_val, 2) * M3_val)
                                 - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                                * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                   + np.power(yd_val, 2)))# end trace
                             - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                                * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))# end trace
                             - (12 * np.power(g2_val, 2) * M2_val
                                * np.power(ytau_val, 2))
                             - (30 * np.power(g2_val, 4) * M2_val)
                             - ((18 / 5) * np.power(g2_val, 2)
                                * np.power(g1_val, 2)
                                * (M1_val + M2_val))
                             - (54 * np.power(g1_val, 4) * M1_val))))

        damu_dt_2l = ((amu_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
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
                                   + np.power(ye_val, 4)))# end trace
                          - (5 * np.power(ymu_val, 2)# Tr(3Yd^2 + Ye^2)
                             * ((3 * (np.power(yb_val, 2)
                                      + np.power(ys_val, 2)
                                      + np.power(yd_val, 2)))
                                + (np.power(ytau_val, 2)
                                   + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2))))# end trace
                          - (6 * np.power(ymu_val, 4))
                          + (((16 * np.power(g3_val, 2))
                              - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                             * (np.power(yb_val, 2)
                                + np.power(ys_val, 2)
                                + np.power(yd_val, 2)))# end trace
                          + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                             * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                + np.power(ye_val, 2)))# end trace
                          + (((12 * np.power(g2_val, 2))
                              - ((6 / 5) * np.power(g1_val, 2)))
                             * np.power(ymu_val, 2))
                          + ((15 / 2) * np.power(g2_val, 4))
                          + ((9 / 5) * np.power(g2_val, 2)
                             * np.power(g1_val, 2))
                          + ((27 / 2) * np.power(g1_val, 4))))
                      + (ymu_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
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
                                             + (ae_val * np.power(ye_val, 3))))))# end trace
                            - (4 * ymu_val * amu_val# Tr(3Yd^2 + Ye^2)
                               * ((3 * (np.power(yb_val, 2)
                                        + np.power(ys_val, 2)
                                        + np.power(yd_val, 2)))
                                  + ((np.power(ytau_val, 2)
                                      + np.power(ymu_val, 2)
                                      + np.power(ye_val, 2)))))# end trace
                            - (6 * np.power(ymu_val, 2)# Tr(3ad*Yd + ae*Ye)
                               * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                                        + (ad_val * yd_val)))
                                  + (atau_val * ytau_val) + (amu_val * ymu_val)
                                  + (ae_val * ye_val)))# end trace
                            - (14 * np.power(ymu_val, 3) * amu_val)
                            + (((32 * np.power(g3_val, 2))
                                - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                               * ((ab_val * yb_val) + (as_val * ys_val)
                                  + (ad_val * yd_val)))# end trace
                            + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                               * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                  + (ae_val * ye_val)))# end trace
                            + (((6 * np.power(g2_val, 2))
                                + ((6 / 5) * np.power(g1_val, 2)))
                               * ymu_val * amu_val)
                            - (((32 * np.power(g3_val, 2) * M3_val)
                                - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                               * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                  + np.power(yd_val, 2)))# end trace
                            - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                               * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2)))# end trace
                            - (12 * np.power(g2_val, 2) * M2_val
                               * np.power(ymu_val, 2))
                            - (30 * np.power(g2_val, 4) * M2_val)
                            - ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                               * (M1_val + M2_val))
                            - (54 * np.power(g1_val, 4) * M1_val))))

        dae_dt_2l = ((ae_val# Tr(3Yd^4 + Yu^2*Yd^2 + Ye^4)
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
                                  + np.power(ye_val, 4)))# end trace
                         - (5 * np.power(ye_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2)
                                     + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (np.power(ytau_val, 2)
                                  + np.power(ymu_val, 2)
                                  + np.power(ye_val, 2))))# end trace
                         - (6 * np.power(ye_val, 4))
                         + (((16 * np.power(g3_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                            * (np.power(yb_val, 2)
                               + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((12 * np.power(g2_val, 2))
                             - ((6 / 5) * np.power(g1_val, 2)))
                            * np.power(ye_val, 2))
                         + ((15 / 2) * np.power(g2_val, 4))
                         + ((9 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2))
                         + ((27 / 2) * np.power(g1_val, 4))))
                     + (ye_val# Tr(6ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + 2ae*Ye^3)
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
                                            + (ae_val * np.power(ye_val, 3))))))# end trace
                           - (4 * ye_val * ae_val# Tr(3Yd^2 + Ye^2)
                              * ((3 * (np.power(yb_val, 2)
                                       + np.power(ys_val, 2)
                                       + np.power(yd_val, 2)))
                                 + ((np.power(ytau_val, 2)
                                     + np.power(ymu_val, 2)
                                     + np.power(ye_val, 2)))))# end trace
                           - (6 * np.power(ye_val, 2)# Tr(3ad*Yd + ae*Ye)
                              * ((3 * ((ab_val * yb_val) + (as_val * ys_val)
                                       + (ad_val * yd_val)))
                                 + (atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           - (14 * np.power(ye_val, 3) * ae_val)
                           + (((32 * np.power(g3_val, 2))
                               - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                              * ((ab_val * yb_val) + (as_val * ys_val)
                                 + (ad_val * yd_val)))# end trace
                           + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                              * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                 + (ae_val * ye_val)))# end trace
                           + (((6 * np.power(g2_val, 2))
                               + ((6 / 5) * np.power(g1_val, 2)))
                              * ye_val * ae_val)
                           - (((32 * np.power(g3_val, 2) * M3_val)
                               - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                              * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                 + np.power(yd_val, 2)))# end trace
                           - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                              * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                 + np.power(ye_val, 2)))# end trace
                           - (12 * np.power(g2_val, 2) * M2_val
                              * np.power(ye_val, 2))
                           - (30 * np.power(g2_val, 4) * M2_val)
                           - ((18 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                              * (M1_val + M2_val))
                           - (54 * np.power(g1_val, 4) * M1_val))))

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
        db_dt_1l = ((b_val# Tr(3Yu^2 + 3Yd^2 + Ye^2)
                     * (((3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2) + np.power(yb_val, 2)
                               + np.power(ys_val, 2) + np.power(yd_val, 2)))
                         + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                         + np.power(ye_val, 2))# end trace
                        - (3 * np.power(g2_val, 2))
                        - ((3 / 5) * np.power(g1_val, 2))))
                    + (mu_val# Tr(6au*Yu + 6ad*Yd + 2ae*Ye)
                       * (((6 * ((at_val * yt_val) + (ac_val * yc_val)
                                 + (au_val * yu_val) + (ab_val * yb_val)
                                 + (as_val * ys_val) + (ad_val * yd_val)))
                           + (2 * ((atau_val * ytau_val) + (amu_val * ymu_val)
                                   + (ae_val * ye_val))))
                          + (6 * np.power(g2_val, 2) * M2_val)
                          + ((6 / 5) * np.power(g1_val, 2) * M1_val))))

        # 2 loop part
        db_dt_2l = ((b_val# Tr(3Yu^4 + 3Yd^4 + 2Yu^2*Yd^2 + Ye^4)
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
                                 + np.power(ye_val, 4)))# end trace
                        + (((16 * np.power(g3_val, 2))
                            + ((4 / 5) * np.power(g1_val, 2)))# Tr(Yu^2)
                           * (np.power(yt_val, 2) + np.power(yc_val, 2)
                              + np.power(yu_val, 2)))# end trace
                        + (((16 * np.power(g3_val, 2))
                            - ((2 / 5) * np.power(g1_val, 2)))# Tr(Yd^2)
                           * (np.power(yb_val, 2) + np.power(ys_val, 2)
                              + np.power(yd_val, 2)))# end trace
                        + ((6 / 5) * np.power(g1_val, 2)# Tr(Ye^2)
                           * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                              + np.power(ye_val, 2)))# end trace
                        + ((15 / 2) * np.power(g2_val, 4))
                        + ((9 / 5) * np.power(g1_val, 2) * np.power(g2_val, 2))
                        + ((207 / 50) * np.power(g1_val, 4))))
                    + (mu_val * (((-12)# Tr(3au*Yu^3 + 3ad*Yd^3 + au*Yd^2*Yu + ad*Yu^2*Yd + ae*Ye^3)
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
                                + (ae_val * np.power(ye_val, 3)))))# end trace
                        + (((32 * np.power(g3_val, 2))
                             + ((8 / 5) * np.power(g1_val, 2)))# Tr(au*Yu)
                            * ((at_val * yt_val) + (ac_val * yc_val)
                               + (au_val * yu_val)))# end trace
                         + (((32 * np.power(g3_val, 2))
                             - ((4 / 5) * np.power(g1_val, 2)))# Tr(ad*Yd)
                            * ((ab_val * yb_val) + (as_val * ys_val)
                               + (ad_val * yd_val)))# end trace
                         + ((12 / 5) * np.power(g1_val, 2)# Tr(ae*Ye)
                            * ((atau_val * ytau_val) + (amu_val * ymu_val)
                               + (ae_val * ye_val)))# end trace
                         - (((32 * np.power(g3_val, 2) * M3_val)
                             + ((8 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yu^2)
                            * (np.power(yt_val, 2) + np.power(yc_val, 2)
                               + np.power(yu_val, 2)))
                         - (((32 * np.power(g3_val, 2) * M3_val)# end trace
                             - ((4 / 5) * np.power(g1_val, 2) * M1_val))# Tr(Yd^2)
                            * (np.power(yb_val, 2) + np.power(ys_val, 2)
                               + np.power(yd_val, 2)))# end trace
                         - ((12 / 5) * np.power(g1_val, 2) * M1_val# Tr(Ye^2)
                            * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (30 * np.power(g2_val, 4) * M2_val)
                         - ((18 / 5) * np.power(g1_val, 2)
                            * np.power(g2_val, 2)
                            * (M1_val + M2_val))
                         - ((414 / 25) * np.power(g1_val, 4) * M1_val))))

        # Total b beta function
        db_dt = (1 / t) * ((loop_fac * db_dt_1l)
                           + (loop_fac_sq * db_dt_2l))

        ##### Scalar squared masses #####
        # Introduce S, S', and sigma terms
        S_val = (mHu_sq_val - mHd_sq_val + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
                 - mL3_sq_val - mL2_sq_val - mL1_sq_val
                 - (2 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
                 + mD3_sq_val + mD2_sq_val + mD1_sq_val
                 + mE3_sq_val + mE2_sq_val + mE1_sq_val)

        # Tr(-(3mHu^2 + mQ^2) * Yu^2 + 4Yu^2 * mU^2 + (3mHd^2 - mQ^2) * Yd^2
        #    - 2Yd^2 * mD^2 + (mHd^2 + mL^2) * Ye^2 - 2Ye^2 * mE^2)
        Spr_val = ((((-1) * ((((3 * mHu_sq_val) + mQ3_sq_val)
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
                            + (np.power(ye_val, 2) * mE1_sq_val))))# end trace
                   + ((((3 / 2) * np.power(g2_val, 2))
                       + ((3 / 10) * np.power(g1_val, 2)))
                      * (mHu_sq_val - mHd_sq_val# Tr(mL^2)
                         - (mL3_sq_val + mL2_sq_val + mL1_sq_val)))# end trace
                   + ((((8 / 3) * np.power(g3_val, 2))
                       + ((3 / 2) * np.power(g2_val, 2))
                       + ((1 / 30) * np.power(g1_val, 2)))# Tr(mQ^2)
                      * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))# end trace
                   - ((((16 / 3) * np.power(g3_val, 2))
                       + ((16 / 15) * np.power(g1_val, 2)))# Tr (mU^2)
                      * (mU3_sq_val + mU2_sq_val + mU1_sq_val))# end trace
                   + ((((8 / 3) * np.power(g3_val, 2))
                      + ((2 / 15) * np.power(g1_val, 2)))# Tr(mD^2)
                      * (mD3_sq_val + mD2_sq_val + mD1_sq_val))# end trace
                   + ((6 / 5) * np.power(g1_val, 2)# Tr(mE^2)
                      * (mE3_sq_val + mE2_sq_val + mE1_sq_val)))# end trace

        sigma1 = ((1 / 5) * np.power(g1_val, 2)
                  * ((3 * (mHu_sq_val + mHd_sq_val))# Tr(mQ^2 + 3mL^2 + 8mU^2 + 2mD^2 + 6mE^2)
                     + mQ3_sq_val + mQ2_sq_val + mQ1_sq_val
                     + (3 * (mL3_sq_val + mL2_sq_val + mL1_sq_val))
                     + (8 * (mU3_sq_val + mU2_sq_val + mU1_sq_val))
                     + (2 * (mD3_sq_val + mD2_sq_val + mD1_sq_val))
                     + (6 * (mE3_sq_val + mE2_sq_val + mE1_sq_val))))# end trace

        sigma2 = (np.power(g2_val, 2)
                  * (mHu_sq_val + mHd_sq_val# Tr(3mQ^2 + mL^2)
                     + (3 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
                     + mL3_sq_val + mL2_sq_val + mL1_sq_val))# end trace

        sigma3 = (np.power(g3_val, 2)# Tr(2mQ^2 + mU^2 + mD^2)
                  * ((2 * (mQ3_sq_val + mQ2_sq_val + mQ1_sq_val))
                     + mU3_sq_val + mU2_sq_val + mU1_sq_val
                     + mD3_sq_val + mD2_sq_val + mD1_sq_val))# end trace

        # 1 loop part of Higgs squared masses
        dmHu_sq_dt_1l = ((6# Tr((mHu^2 + mQ^2) * Yu^2 + Yu^2 * mU^2 + au^2)
                          * (((mHu_sq_val + mQ3_sq_val) * np.power(yt_val, 2))
                             + ((mHu_sq_val + mQ2_sq_val)
                                * np.power(yc_val, 2))
                             + ((mHu_sq_val + mQ1_sq_val)
                                * np.power(yu_val, 2))
                             + (mU3_sq_val * np.power(yt_val, 2))
                             + (mU2_sq_val * np.power(yc_val, 2))
                             + (mU1_sq_val * np.power(yu_val, 2))
                             + np.power(at_val, 2) + np.power(ac_val, 2)
                             + np.power(au_val, 2)))# end trace
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((3 / 5) * np.power(g1_val, 2) * S_val))

        # Tr (6(mHd^2 + mQ^2) * Yd^2 + 6Yd^2*mD^2 + 2(mHd^2 + mL^2) * Ye^2
        #     + 2(Ye^2 * mE^2) + 6ad^2 + 2ae^2)
        dmHd_sq_dt_1l = ((6 * (((mHd_sq_val + mQ3_sq_val)
                                * np.power(yb_val, 2))
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
                                  + np.power(ae_val, 2))))# end trace
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        # 2 loop part of Higgs squared masses
        dmHu_sq_dt_2l = (((-6) # Tr(6(mHu^2 + mQ^2)*Yu^4 + 6Yu^4 * mU^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2 * Yd^2 + Yu^2 * Yd^2 * mU^2 + Yu^2 * Yd^2 * mQ^2 + Yu^2 * Yd^2 * mD^2 + 12au^2 * Yu^2 + ad^2 * Yu^2 + Yd^2 * au^2 + 2ad * Yd * Yu * au)
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
                                     + (yd_val * ad_val * au_val * yu_val)))))# end trace
                         + (((32 * np.power(g3_val, 2))
                             + ((8 / 5) * np.power(g1_val, 2))) # Tr((mHu^2 + mQ^2 + mU^2) * Yu^2 + au^2)
                            * (((mHu_sq_val + mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mHu_sq_val + mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mHu_sq_val + mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))
                               + np.power(at_val, 2) + np.power(ac_val, 2)
                               + np.power(au_val, 2)))# end trace
                         + (32 * np.power(g3_val, 2)
                            * ((2 * np.power(M3_val, 2)# Tr(Yu^2)
                                * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                               - (2 * M3_val# Tr(Yu*au)
                                  * ((yt_val * at_val) + (yc_val * ac_val)
                                     + (yu_val * au_val)))))# end trace
                         + ((8 / 5) * np.power(g1_val, 2)
                            * ((2 * np.power(M1_val, 2)# Tr(Yu^2)
                                * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                               - (2 * M1_val# Tr(Yu*au)
                                  * ((yt_val * at_val) + (yc_val * ac_val)
                                     + (yu_val * au_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M2_val, 2) + np.power(M1_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        dmHd_sq_dt_2l = (((-6) # Tr(6(mHd^2 + mQ^2)*Yd^4 + 6Yd^4 * mD^2 + (mHu^2 + mHd^2 + mQ^2) * Yu^2 * Yd^2 + Yu^2 * Yd^2 * mU^2 + Yu^2 * Yd^2 * mQ^2 + Yu^2 * Yd^2 * mD^2 + 2(mHd^2 + mL^2) * Ye^4 + 2Ye^4 * mE^2 + 12ad^2 * Yd^2 + ad^2 * Yu^2 + Yd^2 * au^2 + 2ad * Yd * Yu * au + 4ae^2 * Ye^2)
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
                                                * np.power(ye_val, 2))))))))# end trace
                         + (((32 * np.power(g3_val, 2))
                             - ((4 / 5) * np.power(g1_val, 2))) # Tr((mHd^2 + mQ^2 + mD^2) * Yd^2 + ad^2)
                            * (((mHu_sq_val + mQ3_sq_val + mD3_sq_val)
                                * np.power(yb_val, 2))
                               + ((mHu_sq_val + mQ2_sq_val + mD2_sq_val)
                                  * np.power(ys_val, 2))
                               + ((mHu_sq_val + mQ1_sq_val + mD1_sq_val)
                                  * np.power(yd_val, 2))
                               + np.power(ab_val, 2) + np.power(as_val, 2)
                               + np.power(ad_val, 2)))# end trace
                         + (32 * np.power(g3_val, 2)
                            * ((2 * np.power(M3_val, 2)# Tr(Yd^2)
                                * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                   + np.power(yd_val, 2)))# end trace
                               - (2 * M3_val # Tr(Yd*ad)
                                  * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))))# end trace
                         - ((4 / 5) * np.power(g1_val, 2)
                            * ((2 * np.power(M1_val, 2)# Tr(Yd^2)
                                * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                   + np.power(yd_val, 2)))# end trace
                               - (2 * M1_val # Tr(Yd*ad)
                                  * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))))# end trace
                         + ((12 / 5) * np.power(g1_val, 2)
                            * (# Tr((mHd^2 + mL^2 + mE^2) * Ye^2 + ae^2)
                               ((mHd_sq_val + mL3_sq_val + mE3_sq_val)
                                * np.power(ytau_val, 2))
                               + ((mHd_sq_val + mL2_sq_val + mE2_sq_val)
                                  * np.power(ymu_val, 2))
                               + ((mHd_sq_val + mL1_sq_val + mE1_sq_val)
                                  * np.power(ye_val, 2))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M2_val, 2) + np.power(M1_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        # Total Higgs squared mass beta functions
        dmHu_sq_dt = (1 / t) * ((loop_fac * dmHu_sq_dt_1l)
                                + (loop_fac_sq * dmHu_sq_dt_2l))

        dmHd_sq_dt = (1 / t) * ((loop_fac * dmHd_sq_dt_1l)
                                + (loop_fac_sq * dmHd_sq_dt_2l))

        # 1 loop parts of scalar squared masses
        # Left squarks
        dmQ3_sq_dt_1l = (((mQ3_sq_val + (2 * mHu_sq_val))
                          * np.power(yt_val, 2))
                         + ((mQ3_sq_val + (2 * mHd_sq_val))
                            * np.power(yb_val, 2))
                         + ((np.power(yt_val, 2) + np.power(yb_val, 2))
                            * mQ3_sq_val)
                         + (2 * np.power(yt_val, 2) * mU3_sq_val)
                         + (2 * np.power(yb_val, 2) * mD3_sq_val)
                         + (2 * np.power(at_val, 2))
                         + (2 * np.power(ab_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((2 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((1 / 5) * np.power(g1_val, 2) * S_val))

        dmQ2_sq_dt_1l = (((mQ2_sq_val + (2 * mHu_sq_val))
                          * np.power(yc_val, 2))
                         + ((mQ2_sq_val + (2 * mHd_sq_val))
                            * np.power(ys_val, 2))
                         + ((np.power(yc_val, 2) + np.power(ys_val, 2))
                            * mQ2_sq_val)
                         + (2 * np.power(yc_val, 2) * mU2_sq_val)
                         + (2 * np.power(ys_val, 2) * mD2_sq_val)
                         + (2 * np.power(ac_val, 2))
                         + (2 * np.power(as_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((2 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((1 / 5) * np.power(g1_val, 2) * S_val))

        dmQ1_sq_dt_1l = (((mQ1_sq_val + (2 * mHu_sq_val))
                          * np.power(yu_val, 2))
                         + ((mQ1_sq_val + (2 * mHd_sq_val))
                            * np.power(yd_val, 2))
                         + ((np.power(yu_val, 2)
                             + np.power(yd_val, 2)) * mQ1_sq_val)
                         + (2 * np.power(yu_val, 2) * mU1_sq_val)
                         + (2 * np.power(yd_val, 2) * mD1_sq_val)
                         + (2 * np.power(au_val, 2))
                         + (2 * np.power(ad_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((2 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((1 / 5) * np.power(g1_val, 2) * S_val))

        # Left leptons
        dmL3_sq_dt_1l = (((mL3_sq_val + (2 * mHd_sq_val))
                          * np.power(ytau_val, 2))
                         + (2 * np.power(ytau_val, 2) * mE3_sq_val)
                         + (np.power(ytau_val, 2) * mL3_sq_val)
                         + (2 * np.power(atau_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        dmL2_sq_dt_1l = (((mL2_sq_val + (2 * mHd_sq_val))
                          * np.power(ymu_val, 2))
                         + (2 * np.power(ymu_val, 2) * mE2_sq_val)
                         + (np.power(ymu_val, 2) * mL2_sq_val)
                         + (2 * np.power(amu_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        dmL1_sq_dt_1l = (((mL1_sq_val + (2 * mHd_sq_val))
                          * np.power(ye_val, 2))
                         + (2 * np.power(ye_val, 2) * mE1_sq_val)
                         + (np.power(ye_val, 2) * mL1_sq_val)
                         + (2 * np.power(ae_val, 2))
                         - (6 * np.power(g2_val, 2) * np.power(M2_val, 2))
                         - ((6 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((3 / 5) * np.power(g1_val, 2) * S_val))

        # Right up-type squarks
        dmU3_sq_dt_1l = ((2 * (mU3_sq_val + (2 * mHd_sq_val))
                          * np.power(yt_val, 2))
                         + (4 * np.power(yt_val, 2) * mQ3_sq_val)
                         + (2 * np.power(yt_val, 2) * mU3_sq_val)
                         + (4 * np.power(at_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((32 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((4 / 5) * np.power(g1_val, 2) * S_val))

        dmU2_sq_dt_1l = ((2 * (mU2_sq_val + (2 * mHd_sq_val))
                          * np.power(yc_val, 2))
                         + (4 * np.power(yc_val, 2) * mQ2_sq_val)
                         + (2 * np.power(yc_val, 2) * mU2_sq_val)
                         + (4 * np.power(ac_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((32 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((4 / 5) * np.power(g1_val, 2) * S_val))

        dmU1_sq_dt_1l = ((2 * (mU1_sq_val + (2 * mHd_sq_val))
                          * np.power(yu_val, 2))
                         + (4 * np.power(yu_val, 2) * mQ1_sq_val)
                         + (2 * np.power(yu_val, 2) * mU1_sq_val)
                         + (4 * np.power(au_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((32 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         - ((4 / 5) * np.power(g1_val, 2) * S_val))

        # Right down-type squarks
        dmD3_sq_dt_1l = ((2 * (mD3_sq_val + (2 * mHd_sq_val))
                          * np.power(yb_val, 2))
                         + (4 * np.power(yb_val, 2) * mQ3_sq_val)
                         + (2 * np.power(yb_val, 2) * mD3_sq_val)
                         + (4 * np.power(ab_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((8 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2) * S_val))

        dmD2_sq_dt_1l = ((2 * (mD2_sq_val + (2 * mHd_sq_val))
                          * np.power(ys_val, 2))
                         + (4 * np.power(ys_val, 2) * mQ2_sq_val)
                         + (2 * np.power(ys_val, 2) * mD2_sq_val)
                         + (4 * np.power(as_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((8 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2) * S_val))

        dmD1_sq_dt_1l = ((2 * (mD1_sq_val + (2 * mHd_sq_val))
                          * np.power(yd_val, 2))
                         + (4 * np.power(yd_val, 2) * mQ1_sq_val)
                         + (2 * np.power(yd_val, 2) * mD1_sq_val)
                         + (4 * np.power(ad_val, 2))
                         - ((32 / 3) * np.power(g3_val, 2)
                            * np.power(M3_val, 2))
                         - ((8 / 15) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((2 / 5) * np.power(g1_val, 2) * S_val))

        # Right leptons
        dmE3_sq_dt_1l = ((2 * (mE3_sq_val + (2 * mHd_sq_val))
                          * np.power(ytau_val, 2))
                         + (4 * np.power(ytau_val, 2) * mL3_sq_val)
                         + (2 * np.power(ytau_val, 2) * mE3_sq_val)
                         + (4 * np.power(atau_val, 2))
                         - ((24 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((6 / 5) * np.power(g1_val, 2) * S_val))

        dmE2_sq_dt_1l = ((2 * (mE2_sq_val + (2 * mHd_sq_val))
                          * np.power(ymu_val, 2))
                         + (4 * np.power(ymu_val, 2) * mL2_sq_val)
                         + (2 * np.power(ymu_val, 2) * mE2_sq_val)
                         + (4 * np.power(amu_val, 2))
                         - ((24 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((6 / 5) * np.power(g1_val, 2) * S_val))

        dmE1_sq_dt_1l = ((2 * (mE1_sq_val + (2 * mHd_sq_val))
                          * np.power(ye_val, 2))
                         + (4 * np.power(ye_val, 2) * mL1_sq_val)
                         + (2 * np.power(ye_val, 2) * mE1_sq_val)
                         + (4 * np.power(ae_val, 2))
                         - ((24 / 5) * np.power(g1_val, 2)
                            * np.power(M1_val, 2))
                         + ((6 / 5) * np.power(g1_val, 2) * S_val))

        # 2 loop parts of scalar squared masses
        # Left squarks
        dmQ3_sq_dt_2l = (((-8) * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                          * np.power(yt_val, 4))
                         - (8 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                            * np.power(yb_val, 4))
                         - (np.power(yt_val, 2)
                            * ((2 * mQ3_sq_val) + (2 * mU3_sq_val)
                               + (4 * mHu_sq_val))# Tr(3Yu^2)
                            * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yb_val, 2)
                            * ((2 * mQ3_sq_val) + (2 * mD3_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (6 * np.power(yt_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (np.power(yb_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(yt_val, 2) * np.power(at_val, 2))
                         - (16 * np.power(yb_val, 2) * np.power(ab_val, 2))
                         - (np.power(at_val, 2)# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yt_val, 2)# Tr(6au^2)
                            * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                                   + np.power(au_val, 2)))# end trace
                         - (at_val * yt_val# Tr(12Yu*au)
                            * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val)))# end trace
                         - (np.power(ab_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(yb_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * ab_val * yb_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((2 / 5) * np.power(g1_val, 2)
                            * ((4 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + (4 * np.power(at_val, 2))
                               - (8 * M1_val * at_val * yt_val)
                               + (8 * np.power(M1_val, 2) * np.power(yt_val, 2))
                               + (2 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                                  * np.power(yb_val, 2))
                               + (2 * np.power(ab_val, 2))
                               - (4 * M1_val * ab_val * yb_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(yb_val, 2))))
                         + ((2 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
                            * (np.power(M3_val, 2) + np.power(M2_val, 2)
                               + (M2_val * M3_val)))
                         + ((32 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((1 / 15) * np.power(g1_val, 2) * sigma1))

        dmQ2_sq_dt_2l = (((-8) * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                          * np.power(yc_val, 4))
                         - (8 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                            * np.power(ys_val, 4))
                         - (np.power(yc_val, 2)
                            * ((2 * mQ2_sq_val) + (2 * mU2_sq_val)
                               + (4 * mHu_sq_val))# Tr(3Yu^2)
                            * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(ys_val, 2)
                            * ((2 * mQ2_sq_val) + (2 * mD2_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (6 * np.power(yc_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (np.power(ys_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(yc_val, 2) * np.power(ac_val, 2))
                         - (16 * np.power(ys_val, 2) * np.power(as_val, 2))
                         - (np.power(ac_val, 2)# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yc_val, 2)# Tr(6au^2)
                            * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                                   + np.power(au_val, 2)))# end trace
                         - (ac_val * yc_val# Tr(12Yu*au)
                            * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val)))# end trace
                         - (np.power(as_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ys_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * as_val * ys_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((2 / 5) * np.power(g1_val, 2)
                            * ((4 * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                                * np.power(yc_val, 2))
                               + (4 * np.power(ac_val, 2))
                               - (8 * M1_val * ac_val * yc_val)
                               + (8 * np.power(M1_val, 2) * np.power(yc_val, 2))
                               + (2 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                                  * np.power(ys_val, 2))
                               + (2 * np.power(as_val, 2))
                               - (4 * M1_val * as_val * ys_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ys_val, 2))))
                         + ((2 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
                            * (np.power(M3_val, 2) + np.power(M2_val, 2)
                               + (M2_val * M3_val)))
                         + ((32 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((1 / 15) * np.power(g1_val, 2) * sigma1))

        dmQ1_sq_dt_2l = (((-8) * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                          * np.power(yu_val, 4))
                         - (8 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                            * np.power(yd_val, 4))
                         - (np.power(yu_val, 2)
                            * ((2 * mQ1_sq_val) + (2 * mU1_sq_val)
                               + (4 * mHu_sq_val))# Tr(3Yu^2)
                            * 3 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yd_val, 2)
                            * ((2 * mQ1_sq_val) + (2 * mD1_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (6 * np.power(yu_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (np.power(yd_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(yu_val, 2) * np.power(au_val, 2))
                         - (16 * np.power(yd_val, 2) * np.power(ad_val, 2))
                         - (np.power(au_val, 2)# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (np.power(yu_val, 2)# Tr(6au^2)
                            * 6 * (np.power(at_val, 2) + np.power(ac_val, 2)
                                   + np.power(au_val, 2)))# end trace
                         - (au_val * yu_val# Tr(12Yu*au)
                            * 12 * ((yt_val * at_val) + (yc_val * ac_val)
                                    + (yu_val * au_val)))# end trace
                         - (np.power(ad_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(yd_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * ad_val * yd_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((2 / 5) * np.power(g1_val, 2)
                            * ((4 * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                                * np.power(yu_val, 2))
                               + (4 * np.power(au_val, 2))
                               - (8 * M1_val * au_val * yu_val)
                               + (8 * np.power(M1_val, 2) * np.power(yu_val, 2))
                               + (2 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                                  * np.power(yd_val, 2))
                               + (2 * np.power(ad_val, 2))
                               - (4 * M1_val * ad_val * yd_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(yd_val, 2))))
                         + ((2 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + (32 * np.power(g3_val, 2) * np.power(g2_val, 2)
                            * (np.power(M3_val, 2) + np.power(M2_val, 2)
                               + (M2_val * M3_val)))
                         + ((32 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((2 / 5) * np.power(g2_val, 2) * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((199 / 75) * np.power(g1_val, 4) * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((1 / 15) * np.power(g1_val, 2) * sigma1))

        # Left leptons
        dmL3_sq_dt_2l = (((-8) * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                          * np.power(ytau_val, 4))
                         - (np.power(ytau_val, 2)
                            * ((2 * mL3_sq_val) + (2 * mE3_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (np.power(ytau_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ytau_val, 2) * np.power(atau_val, 2))
                         - (np.power(atau_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ytau_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * atau_val * ytau_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)
                            * ((2 * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                                * np.power(ytau_val, 2))
                               + (2 * np.power(atau_val, 2))
                               - (4 * M1_val * atau_val
                                  * ytau_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ytau_val, 2))))
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        dmL2_sq_dt_2l = (((-8) * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                          * np.power(ymu_val, 4))
                         - (np.power(ymu_val, 2)
                            * ((2 * mL2_sq_val) + (2 * mE2_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (np.power(ymu_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ymu_val, 2) * np.power(amu_val, 2))
                         - (np.power(amu_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ymu_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * amu_val * ymu_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)
                            * ((2 * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                                * np.power(ymu_val, 2))
                               + (2 * np.power(amu_val, 2))
                               - (4 * M1_val * amu_val
                                  * ymu_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ymu_val, 2))))
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        dmL1_sq_dt_2l = (((-8) * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                          * np.power(ye_val, 4))
                         - (np.power(ye_val, 2)
                            * ((2 * mL1_sq_val) + (2 * mE1_sq_val)
                               + (4 * mHd_sq_val))# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (np.power(ye_val, 2)# Tr(6(mQ^2 + mD^2)*Yd^2 + 2(mL^2 + mE^2)*Ye^2)
                            * ((6 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (2 * (((mL3_sq_val + mE3_sq_val)
                                        * np.power(ytau_val, 2))
                                       + ((mL2_sq_val + mE2_sq_val)
                                          * np.power(ymu_val, 2))
                                       + ((mL1_sq_val + mE1_sq_val)
                                          * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ye_val, 2) * np.power(ae_val, 2))
                         - (np.power(ae_val, 2)# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (np.power(ye_val, 2)# Tr(6ad^2 + 2ae^2)
                            * ((6 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + (2 * (np.power(atau_val, 2)
                                       + np.power(amu_val, 2)
                                       + np.power(ae_val, 2)))))# end trace
                         - (2 * ae_val * ye_val# Tr(6Yd*ad + 2Ye*ae)
                            * ((6 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (2 * ((ytau_val * atau_val)
                                       + (ymu_val * amu_val)
                                       + (ye_val * ae_val)))))# end trace
                         + ((6 / 5) * np.power(g1_val, 2)
                            * ((2 * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                                * np.power(ye_val, 2))
                               + (2 * np.power(ae_val, 2))
                               - (4 * M1_val * ae_val
                                  * ye_val)
                               + (4 * np.power(M1_val, 2)
                                  * np.power(ye_val, 2))))
                         - ((6 / 5) * np.power(g1_val, 2) * Spr_val)
                         + (33 * np.power(g2_val, 4) * np.power(M2_val, 2))
                         + ((18 / 5) * np.power(g2_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M1_val, 2) + np.power(M2_val, 2)
                               + (M1_val * M2_val)))
                         + ((621 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + (3 * np.power(g2_val, 2) * sigma2)
                         + ((3 / 5) * np.power(g1_val, 2) * sigma1))

        # Right up-type squarks
        dmU3_sq_dt_2l = (((-8) * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                          * np.power(yt_val, 4))
                         - (4 * (mU3_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ3_sq_val) + mD3_sq_val)
                            * np.power(yb_val, 2) * np.power(yt_val, 2))
                         - (np.power(yt_val, 2)
                            * ((2 * mQ3_sq_val) + (2 * mU3_sq_val)
                               + (4 * mHu_sq_val))# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (12 * np.power(yt_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (16 * np.power(yt_val, 2) * np.power(at_val, 2))
                         - (16 * at_val * ab_val * yb_val * yt_val)
                         - (12 * ((np.power(at_val, 2)# Tr(Yu^2)
                                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                      + np.power(yu_val, 2)))# end trace
                                  + (np.power(yt_val, 2) # Tr(au^2)
                                     * (np.power(at_val, 2)
                                        + np.power(ac_val, 2)
                                        + np.power(au_val, 2)))# end trace
                                  + (at_val * yt_val * 2# Tr(Yu*au)
                                     * ((yt_val * at_val) + (yc_val * ac_val)
                                        + (yu_val * au_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ3_sq_val + mHu_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + (2 * np.power(at_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yt_val, 2))
                                   - (M2_val * at_val * yt_val)))
                         - ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yt_val, 2))
                                   - (M1_val * at_val * yt_val)))
                         - ((8 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((512 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((3424 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((16 / 15) * np.power(g1_val, 2) * sigma1))

        dmU2_sq_dt_2l = (((-8) * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                          * np.power(yc_val, 4))
                         - (4 * (mU2_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ2_sq_val)
                                 + mD2_sq_val)
                            * np.power(ys_val, 2) * np.power(yc_val, 2))
                         - (np.power(yc_val, 2)
                            * ((2 * mQ2_sq_val) + (2 * mU2_sq_val)
                               + (4 * mHu_sq_val))# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (12 * np.power(yc_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (16 * np.power(yc_val, 2) * np.power(ac_val, 2))
                         - (16 * ac_val * as_val * ys_val * yc_val)
                         - (12 * ((np.power(ac_val, 2)# Tr(Yu^2)
                                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                      + np.power(yu_val, 2)))# end trace
                                  + (np.power(yc_val, 2) # Tr(au^2)
                                     * (np.power(at_val, 2)
                                        + np.power(ac_val, 2)
                                        + np.power(au_val, 2)))# end trace
                                  + (ac_val * yc_val * 2# Tr(Yu*au)
                                     * ((yt_val * at_val) + (yc_val * ac_val)
                                        + (yu_val * au_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ2_sq_val + mHu_sq_val + mU2_sq_val)
                                * np.power(yc_val, 2))
                               + (2 * np.power(ac_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yc_val, 2))
                                   - (M2_val * ac_val * yc_val)))
                         - ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yc_val, 2))
                                   - (M1_val * ac_val * yc_val)))
                         - ((8 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((512 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((3424 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((16 / 15) * np.power(g1_val, 2) * sigma1))

        dmU1_sq_dt_2l = (((-8) * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                          * np.power(yu_val, 4))
                         - (4 * (mU1_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ1_sq_val)
                                 + mD1_sq_val)
                            * np.power(yd_val, 2) * np.power(yu_val, 2))
                         - (np.power(yu_val, 2)
                            * ((2 * mQ1_sq_val) + (2 * mU1_sq_val)
                               + (4 * mHu_sq_val))# Tr(6Yu^2)
                            * 6 * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                   + np.power(yu_val, 2)))# end trace
                         - (12 * np.power(yu_val, 2)# Tr((mQ^2 + mU^2)*Yu^2)
                            * (((mQ3_sq_val + mU3_sq_val)
                                * np.power(yt_val, 2))
                               + ((mQ2_sq_val + mU2_sq_val)
                                  * np.power(yc_val, 2))
                               + ((mQ1_sq_val + mU1_sq_val)
                                  * np.power(yu_val, 2))))# end trace
                         - (16 * np.power(yu_val, 2) * np.power(au_val, 2))
                         - (16 * au_val * ad_val * yd_val * yu_val)
                         - (12 * ((np.power(au_val, 2)# Tr(Yu^2)
                                   * (np.power(yt_val, 2) + np.power(yc_val, 2)
                                      + np.power(yu_val, 2)))# end trace
                                  + (np.power(yu_val, 2) # Tr(au^2)
                                     * (np.power(at_val, 2)
                                        + np.power(ac_val, 2)
                                        + np.power(au_val, 2)))# end trace
                                  + (au_val * yu_val * 2# Tr(Yu*au)
                                     * ((yt_val * at_val) + (yc_val * ac_val)
                                        + (yu_val * au_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ1_sq_val + mHu_sq_val + mU1_sq_val)
                                * np.power(yu_val, 2))
                               + (2 * np.power(au_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yu_val, 2))
                                   - (M2_val * au_val * yu_val)))
                         - ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yu_val, 2))
                                   - (M1_val * au_val * yu_val)))
                         - ((8 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((512 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((3424 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((16 / 15) * np.power(g1_val, 2) * sigma1))

        # Right down-type squarks
        dmD3_sq_dt_2l = (((-8) * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                          * np.power(yb_val, 4))
                         - (4 * (mU3_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ3_sq_val)
                                 + mD3_sq_val) * np.power(yb_val, 2)
                            * np.power(yt_val, 2))
                         - (np.power(yb_val, 2)
                            * (2 * (mD3_sq_val + mQ3_sq_val
                                    + (2 * mHd_sq_val)))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(yb_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (((mL3_sq_val + mE3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mL2_sq_val + mE2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mL1_sq_val + mE1_sq_val)
                                     * np.power(ye_val, 2)))# end trace
                               ))
                         - (16 * np.power(yb_val, 2) * np.power(ab_val, 2))
                         - (16 * at_val * ab_val * yb_val * yt_val)
                         - (4 * np.power(ab_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(yb_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - (8 * ab_val * yb_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((6 * np.power(g2_val, 2))
                             + ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ3_sq_val + mHd_sq_val + mD3_sq_val)
                                * np.power(yb_val, 2))
                               + (2 * np.power(ab_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yb_val, 2))
                                   - (M2_val * ab_val * yb_val)))
                         + ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yb_val, 2))
                                   - (M1_val * ab_val * yb_val)))
                         + ((4 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((128 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((808 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((4 / 15) * np.power(g1_val, 2) * sigma1))

        dmD2_sq_dt_2l = (((-8) * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                          * np.power(ys_val, 4))
                         - (4 * (mU2_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ2_sq_val)
                                 + mD2_sq_val) * np.power(ys_val, 2)
                            * np.power(yc_val, 2))
                         - (np.power(ys_val, 2)
                            * (2 * (mD2_sq_val + mQ2_sq_val
                                    + (2 * mHd_sq_val)))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ys_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (((mL3_sq_val + mE3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mL2_sq_val + mE2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mL1_sq_val + mE1_sq_val)
                                     * np.power(ye_val, 2)))# end trace
                               ))
                         - (16 * np.power(ys_val, 2) * np.power(as_val, 2))
                         - (16 * ac_val * as_val * ys_val * yc_val)
                         - (4 * np.power(as_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(ys_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - (8 * as_val * ys_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((6 * np.power(g2_val, 2))
                             + ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ2_sq_val + mHd_sq_val + mD2_sq_val)
                                * np.power(ys_val, 2))
                               + (2 * np.power(as_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(ys_val, 2))
                                   - (M2_val * as_val * ys_val)))
                         + ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(ys_val, 2))
                                   - (M1_val * as_val * ys_val)))
                         + ((4 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((128 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((808 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((4 / 15) * np.power(g1_val, 2) * sigma1))

        dmD1_sq_dt_2l = (((-8) * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                          * np.power(yd_val, 4))
                         - (4 * (mU1_sq_val + mHu_sq_val + mHd_sq_val
                                 + (2 * mQ1_sq_val)
                                 + mD1_sq_val) * np.power(yd_val, 2)
                            * np.power(yu_val, 2))
                         - (np.power(yd_val, 2)
                            * (2 * (mD1_sq_val + mQ1_sq_val
                                    + (2 * mHd_sq_val)))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(yd_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + (((mL3_sq_val + mE3_sq_val)
                                   * np.power(ytau_val, 2))
                                  + ((mL2_sq_val + mE2_sq_val)
                                     * np.power(ymu_val, 2))
                                  + ((mL1_sq_val + mE1_sq_val)
                                     * np.power(ye_val, 2)))# end trace
                               ))
                         - (16 * np.power(yd_val, 2) * np.power(ad_val, 2))
                         - (16 * au_val * ad_val * yd_val * yu_val)
                         - (4 * np.power(ad_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         - (4 * np.power(yd_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + np.power(atau_val, 2) + np.power(amu_val, 2)
                               + np.power(ae_val, 2)))# end trace
                         - (8 * ad_val * yd_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + np.power(ytau_val, 2) + np.power(ymu_val, 2)
                               + np.power(ye_val, 2)))# end trace
                         + (((6 * np.power(g2_val, 2))
                             + ((2 / 5) * np.power(g1_val, 2)))
                            * ((2 * (mQ1_sq_val + mHd_sq_val + mD1_sq_val)
                                * np.power(yd_val, 2))
                               + (2 * np.power(ad_val, 2))))
                         + (12 * np.power(g2_val, 2)
                            * 2 * ((np.power(M2_val, 2) * np.power(yd_val, 2))
                                   - (M2_val * ad_val * yd_val)))
                         + ((4 / 5) * np.power(g1_val, 2)
                            * 2 * ((np.power(M1_val, 2) * np.power(yd_val, 2))
                                   - (M1_val * ad_val * yd_val)))
                         + ((4 / 5) * np.power(g1_val, 2) * Spr_val)
                         - ((128 / 3) * np.power(g3_val, 4)
                            * np.power(M3_val, 2))
                         + ((128 / 45) * np.power(g3_val, 2)
                            * np.power(g1_val, 2)
                            * (np.power(M3_val, 2) + np.power(M1_val, 2)
                               + (M3_val * M1_val)))
                         + ((808 / 75) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((16 / 3) * np.power(g3_val, 2) * sigma3)
                         + ((4 / 15) * np.power(g1_val, 2) * sigma1))

        # Right leptons
        dmE3_sq_dt_2l = (((-8) * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                         * np.power(ytau_val, 4))
            - (np.power(ytau_val, 2)
               * ((2 * mL3_sq_val) + (2 * mE3_sq_val)
                  + (4 * mHd_sq_val))# Tr(6Yd^2 + 2Ye^2)
               * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  + (2 * (np.power(ytau_val, 2) + np.power(ymu_val, 2)
                          + np.power(ye_val, 2)))))# end trace
            - (4 * np.power(ytau_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
               * ((3 * (((mQ3_sq_val + mD3_sq_val) * np.power(yb_val, 2))
                        + ((mQ2_sq_val + mD2_sq_val) * np.power(ys_val, 2))
                        + ((mQ1_sq_val + mD1_sq_val) * np.power(yd_val, 2))))
                  + ((((mL3_sq_val + mE3_sq_val) * np.power(ytau_val, 2))
                      + ((mL2_sq_val + mE2_sq_val) * np.power(ymu_val, 2))
                      + ((mL1_sq_val + mE1_sq_val) * np.power(ye_val, 2))))# end trace
                  ))
            - (16 * np.power(ytau_val, 2) * np.power(atau_val, 2))
            - (4 * np.power(atau_val, 2)# Tr(3Yd^2 + Ye^2)
               * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                        + np.power(yd_val, 2)))
                  + ((np.power(ytau_val, 2) + np.power(ymu_val, 2)
                      + np.power(ye_val, 2)))))# end trace
            - (4 * np.power(ytau_val, 2) # Tr(3ad^2 + ae^2)
               * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                        + np.power(ad_val, 2)))
                  + ((np.power(atau_val, 2) + np.power(amu_val, 2)
                      + np.power(ae_val, 2)))))# end trace
            - (8 * atau_val * ytau_val # Tr(3Yd * ad + Ye * ae)
               * ((3 * ((yb_val * ab_val) + (ys_val * as_val)
                         + (yd_val * ad_val)))
                  + (((ytau_val * atau_val) + (ymu_val * amu_val)
                      + (ye_val * ae_val)))))# end trace
            + (((6 * np.power(g2_val, 2)) - (6 / 5) * np.power(g1_val, 2))
               * ((2 * (mL3_sq_val + mHd_sq_val + mE3_sq_val)
                   * np.power(ytau_val, 2))
                  + (2 * np.power(atau_val, 2))))
            + (12 * np.power(g2_val, 2) * 2
               * ((np.power(M2_val, 2) * np.power(ytau_val, 2))
                  - (M2_val * atau_val * ytau_val)))
            - ((12 / 5) * np.power(g1_val, 2) * 2
               * ((np.power(M1_val, 2) * np.power(ytau_val, 2))
                  - (M1_val * atau_val * ytau_val)))
            + ((12 / 5) * np.power(g1_val, 2) * Spr_val)
            + ((2808 / 25) * np.power(g1_val, 4) * np.power(M1_val, 2))
            + ((12 / 5) * np.power(g1_val, 2) * sigma1))

        dmE2_sq_dt_2l = (((-8) * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                          * np.power(ymu_val, 4))
                         - (np.power(ymu_val, 2)
                            * ((2 * mL2_sq_val) + (2 * mE2_sq_val)
                               + (4 * mHd_sq_val))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ymu_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + ((((mL3_sq_val + mE3_sq_val)
                                    * np.power(ytau_val, 2))
                                   + ((mL2_sq_val + mE2_sq_val)
                                      * np.power(ymu_val, 2))
                                   + ((mL1_sq_val + mE1_sq_val)
                                      * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ymu_val, 2) * np.power(amu_val, 2))
                         - (4 * np.power(amu_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + ((np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ymu_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + ((np.power(atau_val, 2) + np.power(amu_val, 2)
                                   + np.power(ae_val, 2)))))# end trace
                         - (8 * amu_val * ymu_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (((ytau_val * atau_val) + (ymu_val * amu_val)
                                   + (ye_val * ae_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - (6 / 5) * np.power(g1_val, 2))
                            * ((2 * (mL2_sq_val + mHd_sq_val + mE2_sq_val)
                                * np.power(ymu_val, 2))
                               + (2 * np.power(amu_val, 2))))
                         + (12 * np.power(g2_val, 2) * 2
                            * ((np.power(M2_val, 2) * np.power(ymu_val, 2))
                               - (M2_val * amu_val * ymu_val)))
                         - ((12 / 5) * np.power(g1_val, 2) * 2
                            * ((np.power(M1_val, 2) * np.power(ymu_val, 2))
                               - (M1_val * amu_val * ymu_val)))
                         + ((12 / 5) * np.power(g1_val, 2) * Spr_val)
                         + ((2808 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((12 / 5) * np.power(g1_val, 2) * sigma1))

        dmE1_sq_dt_2l = (((-8) * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                          * np.power(ye_val, 4))
                         - (np.power(ye_val, 2)
                            * ((2 * mL1_sq_val) + (2 * mE1_sq_val)
                               + (4 * mHd_sq_val))# Tr(6Yd^2 + 2Ye^2)
                            * ((6 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + (2 * (np.power(ytau_val, 2)
                                       + np.power(ymu_val, 2)
                                       + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ye_val, 2) # Tr(3(mQ^2 + mD^2) * Yd^2 + (mL^2 + mE^2) * Ye^2)
                            * ((3 * (((mQ3_sq_val + mD3_sq_val)
                                      * np.power(yb_val, 2))
                                     + ((mQ2_sq_val + mD2_sq_val)
                                        * np.power(ys_val, 2))
                                     + ((mQ1_sq_val + mD1_sq_val)
                                        * np.power(yd_val, 2))))
                               + ((((mL3_sq_val + mE3_sq_val)
                                    * np.power(ytau_val, 2))
                                   + ((mL2_sq_val + mE2_sq_val)
                                      * np.power(ymu_val, 2))
                                   + ((mL1_sq_val + mE1_sq_val)
                                      * np.power(ye_val, 2))))# end trace
                               ))
                         - (16 * np.power(ye_val, 2) * np.power(ae_val, 2))
                         - (4 * np.power(ae_val, 2)# Tr(3Yd^2 + Ye^2)
                            * ((3 * (np.power(yb_val, 2) + np.power(ys_val, 2)
                                     + np.power(yd_val, 2)))
                               + ((np.power(ytau_val, 2) + np.power(ymu_val, 2)
                                   + np.power(ye_val, 2)))))# end trace
                         - (4 * np.power(ye_val, 2) # Tr(3ad^2 + ae^2)
                            * ((3 * (np.power(ab_val, 2) + np.power(as_val, 2)
                                     + np.power(ad_val, 2)))
                               + ((np.power(atau_val, 2) + np.power(amu_val, 2)
                                   + np.power(ae_val, 2)))))# end trace
                         - (8 * ae_val * ye_val # Tr(3Yd * ad + Ye * ae)
                            * ((3 * ((yb_val * ab_val) + (ys_val * as_val)
                                     + (yd_val * ad_val)))
                               + (((ytau_val * atau_val) + (ymu_val * amu_val)
                                   + (ye_val * ae_val)))))# end trace
                         + (((6 * np.power(g2_val, 2))
                             - (6 / 5) * np.power(g1_val, 2))
                            * ((2 * (mL1_sq_val + mHd_sq_val + mE1_sq_val)
                                * np.power(ye_val, 2))
                               + (2 * np.power(ae_val, 2))))
                         + (12 * np.power(g2_val, 2) * 2
                            * ((np.power(M2_val, 2) * np.power(ye_val, 2))
                               - (M2_val * ae_val * ye_val)))
                         - ((12 / 5) * np.power(g1_val, 2) * 2
                            * ((np.power(M1_val, 2) * np.power(ye_val, 2))
                               - (M1_val * ae_val * ye_val)))
                         + ((12 / 5) * np.power(g1_val, 2) * Spr_val)
                         + ((2808 / 25) * np.power(g1_val, 4)
                            * np.power(M1_val, 2))
                         + ((12 / 5) * np.power(g1_val, 2) * sigma1))

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
        # 1 loop part
        # dtanb_dt_1l = 3 * (np.power(yt_val, 2) - np.power(yb_val, 2))

        # 2 loop part
        # dtanb_dt_2l = (((-9) * (np.power(yt_val, 4) - np.power(yb_val, 4)))
        #               + (6 * np.power(yt_val, 2)
        #                  * (((8 / 3) * np.power(g3_val, 2))
        #                     + ((6 / 45) * np.power(g1_val, 2))))
        #               - (6 * np.power(yb_val, 2)
        #                  * (((8 / 3) * np.power(g3_val, 2))
        #                     - ((3 / 45) * np.power(g1_val, 2))))
        #               - (3 * (np.power(yt_val, 2) - np.power(yb_val, 2))
        #                  * (((1 / np.sqrt(2))
        #                      * (((3 / 5) * np.power(g1_val, 2))
        #                         + np.power(g2_val, 2)))
        #                     + np.power(g2_val, 2))))

        # Total beta function for tanb
        # dtanb_dt = (tanb_val / t) * ((loop_fac * dtanb_dt_1l)
        #                             + (loop_fac_sq * dtanb_dt_2l))


        # Collect all for return
        dxdt = [dg1_dt, dg2_dt, dg3_dt, dM1_dt, dM2_dt, dM3_dt, dmu_dt, dyt_dt,
                dyc_dt, dyu_dt, dyb_dt, dys_dt, dyd_dt, dytau_dt, dymu_dt,
                dye_dt, dat_dt, dac_dt, dau_dt, dab_dt, das_dt, dad_dt,
                datau_dt, damu_dt, dae_dt, dmHu_sq_dt, dmHd_sq_dt,
                dmQ1_sq_dt, dmQ2_sq_dt, dmQ3_sq_dt, dmL1_sq_dt, dmL2_sq_dt,
                dmL3_sq_dt, dmU1_sq_dt, dmU2_sq_dt, dmU3_sq_dt, dmD1_sq_dt,
                dmD2_sq_dt, dmD3_sq_dt, dmE1_sq_dt, dmE2_sq_dt, dmE3_sq_dt,
                db_dt]
        return dxdt

    # Set up domains for solve_ivp
    t_span = np.array([weak_scale * 0.999999, my_QGUT*1.0000001])

    # Now solve
    sol = solve_ivp(my_odes_2, t_span, weak_BCs, t_eval = t_vals,
                    dense_output=True, method='DOP853', atol=1e-9, rtol=1e-9)

    #t = sol.t
    x = sol.y
    return x


def my_radcorr_calc(myQ, vHiggs_wk, mu_wk, beta_wk, yt_wk, yb_wk, ytau_wk,
                    g1_wk, g2_wk, g3_wk, mQ3_sq_wk, mQ2_sq_wk, mQ1_sq_wk,
                    mL3_sq_wk, mL2_sq_wk, mL1_sq_wk, mU3_sq_wk, mU2_sq_wk,
                    mU1_sq_wk, mD3_sq_wk, mD2_sq_wk, mD1_sq_wk, mE3_sq_wk,
                    mE2_sq_wk, mE1_sq_wk, M1_wk, M2_wk, M3_wk,
                    mHu_sq_wk, mHd_sq_wk, at_wk, ab_wk, atau_wk):
    """
    Compute 1-loop and some 2-loop radiative corrections to Higgs scalar
    potential for evaluation of b=B*mu soft SUSY-breaking bilinear parameter
    boundary value at weak scale.

    Parameters
    ----------
    myQ: Float.
        DESCRIPTION.
    vHiggs_wk : Float.
        DESCRIPTION.
    mu_wk : Float.
        DESCRIPTION.
    beta_wk : Float.
        DESCRIPTION.
    yt_wk : Float.
        DESCRIPTION.
    yb_wk : Float.
        DESCRIPTION.
    ytau_wk : Float.
        DESCRIPTION.
    g1_wk : Float.
        DESCRIPTION.
    g2_wk : Float.
        DESCRIPTION.
    g3_wk : Float.
        DESCRIPTION.
    mQ3_sq_wk : Float.
        DESCRIPTION.
    mQ2_sq_wk : Float.
        DESCRIPTION.
    mQ1_sq_wk : Float.
        DESCRIPTION.
    mL3_sq_wk : Float.
        DESCRIPTION.
    mL2_sq_wk : Float.
        DESCRIPTION.
    mL1_sq_wk : Float.
        DESCRIPTION.
    mU3_sq_wk : Float.
        DESCRIPTION.
    mU2_sq_wk : Float.
        DESCRIPTION.
    mU1_sq_wk : Float.
        DESCRIPTION.
    mD3_sq_wk : Float.
        DESCRIPTION.
    mD2_sq_wk : Float.
        DESCRIPTION.
    mD1_sq_wk : Float.
        DESCRIPTION.
    mE3_sq_wk : Float.
        DESCRIPTION.
    mE2_sq_wk : Float.
        DESCRIPTION.
    mE1_sq_wk : Float.
        DESCRIPTION.
    M1_wk : Float.
        DESCRIPTION.
    M2_wk : Float.
        DESCRIPTION.
    M3_wk : Float.
        DESCRIPTION.
    mHu_sq_wk : Float.
        DESCRIPTION.
    mHd_sq_wk : Float.
        DESCRIPTION.
    at_wk : Float.
        DESCRIPTION.
    ab_wk : Float.
        DESCRIPTION.
    atau_wk : Float.
        DESCRIPTION.

    Returns
    -------
    my_radcorrs : Array of floats.
        DESCRIPTION.

    """
    gpr_wk = g1_wk * np.sqrt(3. / 5.)

    ##### Fundamental equations: #####

    def logfunc(mass, Q_renorm_sq=np.power(myQ, 2)):
        """
        Return F = m^2 * (ln(m^2 / Q^2) - 1), where input mass term is linear.

        Parameters
        ----------
        mass : Float.
            Input mass to be evaluated.
        Q_renorm_sq : Float.
            Squared renormalization scale, default value is weak scale, 
            MSUSY=sqrt(mt1(MSUSY) * mt2(MSUSY)) from SoftSUSY.

        Returns
        -------
        myf : Float.
            Return F = m^2 * (ln(m^2 / Q^2) - 1),
            where input mass term is linear.

        """
        myf = np.power(mass, 2) * (np.log((np.power(mass, 2))
                                          / (Q_renorm_sq)) - 1)
        return myf


    def logfunc2(masssq, Q_renorm_sq=np.power(myQ, 2)):
        """
        Return F = m^2 * (ln(m^2 / Q^2) - 1), where input mass term is
        quadratic.

        Parameters
        ----------
        mass : Float.
            Input mass to be evaluated.
        Q_renorm_sq : Float.
            Squared renormalization scale, default value is weak scale, 
            MSUSY=sqrt(mt1(MSUSY) * mt2(MSUSY)) from SoftSUSY.

        Returns
        -------
        myf : Float.
            Return F = m^2 * (ln(m^2 / Q^2) - 1),
            where input mass term is quadratic.

        """
        myf2 = masssq * (np.log((masssq) / (Q_renorm_sq)) - 1)
        return myf2


    sinsqb = np.power(np.sin(beta_wk), 2)
    cossqb = np.power(np.cos(beta_wk), 2)
    v_higgs_u = vHiggs_wk * np.sqrt(sinsqb)
    v_higgs_d = vHiggs_wk * np.sqrt(cossqb)
    tan_th_w = gpr_wk / g2_wk
    theta_w = np.arctan(tan_th_w)
    sinsq_th_w = np.power(np.sin(theta_w), 2)
    cos2b = np.cos(2 * beta_wk)
    sin2b = np.sin(2 * beta_wk)
    gz_sq = (np.power(g2_wk, 2) + np.power(gpr_wk, 2)) / 8

    ##### Mass relations: #####
    # W-boson tree-level running squared mass
    m_w_sq = (np.power(g2_wk, 2) / 2) * np.power(vHiggs_wk, 2)
    # Z-boson tree-level running squared mass
    mz_q_sq = np.power(vHiggs_wk, 2) * ((np.power(g2_wk, 2)
                                         + np.power(gpr_wk, 2)) / 2)
    # Higgs psuedoscalar tree-level running squared mass (computed w/ 2-loop
    # level parameters)
    mA0sq = 2 * np.power(np.abs(mu_wk), 2) + mHu_sq_wk + mHd_sq_wk
    # Top quark tree-level running mass
    mymt = yt_wk * v_higgs_u
    mymtsq = np.power(mymt, 2)
    # Bottom quark tree-level running mass
    mymb = yb_wk * v_higgs_d
    mymbsq = np.power(mymb, 2)
    # Tau tree-level running mass
    mymtau = ytau_wk * v_higgs_d
    mymtausq = np.power(mymtau, 2)

    # Sneutrino running masses
    mselecneutsq = mL1_sq_wk + (0.5 * mz_q_sq * cos2b)
    msmuneutsq = mL2_sq_wk + (0.5 * mz_q_sq * cos2b)
    mstauneutsq = mL3_sq_wk + (0.5 * mz_q_sq * cos2b)

    # Tree-level charged Higgs doublet running squared mass.
    mH_pmsq = mA0sq + m_w_sq

    # Some terms for mass eigenstate-basis eigenvalue computations.
    DeltauL = (0.5 - ((2.0 / 3.0) * sinsq_th_w)) * cos2b * mz_q_sq
    DeltauR = (((2.0 / 3.0) * sinsq_th_w)) * cos2b * mz_q_sq
    DeltadL = (-0.5 + ((1.0 / 3.0) * sinsq_th_w)) * cos2b * mz_q_sq
    DeltadR = (-1.0 / 3.0) * sinsq_th_w * cos2b * mz_q_sq
    DeltaeL = (-0.5 + sinsq_th_w) * cos2b * mz_q_sq
    DeltaeR = (-1.0 * sinsq_th_w) * cos2b * mz_q_sq

    # Stop mass eigenstate eigenvalues
    m_stop_1sq = (0.5)\
        * ((2 * mymtsq) + mQ3_sq_wk + mU3_sq_wk + DeltauL + DeltauR
           - np.sqrt(np.power((mQ3_sq_wk - mU3_sq_wk + DeltauL - DeltauR), 2)
                     + (4 * np.power((at_wk * v_higgs_u)
                                      - (v_higgs_d * yt_wk * mu_wk), 2))))
    m_stop_2sq = (0.5)\
        * ((2 * mymtsq) + mQ3_sq_wk + mU3_sq_wk + DeltauL + DeltauR
           + np.sqrt(np.power((mQ3_sq_wk - mU3_sq_wk + DeltauL - DeltauR), 2)
                     + (4 * np.power((at_wk * v_higgs_u)
                                      - (v_higgs_d * yt_wk * mu_wk), 2))))

    # Sbottom mass eigenstate eigenvalues
    m_sbot_1sq = (0.5)\
        * ((2 * mymbsq) + mQ3_sq_wk + mD3_sq_wk + DeltadL + DeltadR
           - np.sqrt(np.power((mQ3_sq_wk - mD3_sq_wk + DeltadL - DeltadR), 2)
                     + (4 * np.power((ab_wk * v_higgs_d)
                                     - (v_higgs_u * yb_wk * mu_wk), 2))))
    m_sbot_2sq = (0.5)\
        * ((2 * mymbsq) + mQ3_sq_wk + mD3_sq_wk + DeltadL + DeltadR
           + np.sqrt(np.power((mQ3_sq_wk - mD3_sq_wk + DeltadL - DeltadR), 2)
                     + (4 * np.power((ab_wk * v_higgs_d)
                                     - (v_higgs_u * yb_wk * mu_wk), 2))))

    # Stau mass eigenstate eigenvalues
    m_stau_1sq = (0.5)\
        * ((2 * mymtausq) + mL3_sq_wk + mE3_sq_wk + DeltaeL + DeltaeR
           - np.sqrt(np.power((mL3_sq_wk - mE3_sq_wk + DeltaeL - DeltaeR), 2)
                     + (4 * np.power((atau_wk * v_higgs_d)
                                     - (v_higgs_u * ytau_wk * mu_wk), 2))))
    m_stau_2sq = (0.5)\
        * ((2 * mymtausq) + mL3_sq_wk + mE3_sq_wk + DeltaeL + DeltaeR
           + np.sqrt(np.power((mL3_sq_wk - mE3_sq_wk + DeltaeL - DeltaeR), 2)
                     + (4 * np.power((atau_wk * v_higgs_d)
                                     - (v_higgs_u * ytau_wk * mu_wk), 2))))

    # Chargino mass eigenstate eigenvalues
    msC1sq = (0.5)\
        * ((np.abs(M2_wk) ** 2) + (np.abs(mu_wk) ** 2) + (2.0 * m_w_sq)
           - np.sqrt(np.power(((np.abs(M2_wk) ** 2) + (np.abs(mu_wk) ** 2)
                               + (2.0 * m_w_sq)), 2)
                     - (4.0 * np.power(np.abs((mu_wk * M2_wk)
                                              - (m_w_sq * sin2b)), 2))))
    msC2sq = (0.5)\
        * ((np.abs(M2_wk) ** 2) + (np.abs(mu_wk) ** 2) + (2.0 * m_w_sq)
           + np.sqrt(np.power(((np.abs(M2_wk) ** 2) + (np.abs(mu_wk) ** 2)
                               + (2.0 * m_w_sq)), 2)
                     - (4.0 * np.power(np.abs((mu_wk * M2_wk)
                                              - (m_w_sq * sin2b)), 2))))

    # Neutralino mass eigenstate eigenvalues
    neut_mass_mat = np.array([[M1_wk, 0, (-1) * gpr_wk * v_higgs_d,
                               gpr_wk * v_higgs_u],
                              [0, M2_wk, g2_wk * v_higgs_d,
                               (-1) * g2_wk * v_higgs_u],
                              [(-1) * gpr_wk * v_higgs_d,
                               g2_wk * v_higgs_d, 0, (-1) * mu_wk],
                              [gpr_wk * v_higgs_u,
                               (-1) * g2_wk * v_higgs_u, (-1) * mu_wk, 0]])
    my_neut_eigvals, my_neut_eigvecs = np.linalg.eig(neut_mass_mat)
    sorted_eigvals = sorted(my_neut_eigvals, key=abs)
    msN1 = sorted_eigvals[0]
    msN2 = sorted_eigvals[1]
    msN3 = sorted_eigvals[2]
    msN4 = sorted_eigvals[3]

    # Neutral Higgs doublet mass eigenstate running squared masses
    mh0sq = (0.5)\
        * (mA0sq + mz_q_sq
           - np.sqrt(np.power((mA0sq - mz_q_sq), 2)
                     + (4.0 * mz_q_sq * mA0sq * (np.power(sin2b, 2)))))
    mH0sq = (0.5)\
        * (mA0sq + mz_q_sq
           + np.sqrt(np.power((mA0sq - mz_q_sq), 2)
                     + (4.0 * mz_q_sq * mA0sq * (np.power(sin2b, 2)))))

    ##### Radiative corrections in stop squark sector #####
    delta_stop = ((1 / 2) - (4 / 3) * sinsq_th_w) * 2 \
        * (((mQ3_sq_wk - mU3_sq_wk) / 2)
           + (mz_q_sq * cos2b * ((1 / 4) - (2 / 3) * sinsq_th_w)))
    stopuu_num = np.power(at_wk, 2) - (2 * gz_sq * delta_stop)
    stopdd_num = (np.power(yt_wk, 2) * np.power(mu_wk, 2))\
        + (2 * gz_sq * delta_stop)
    sigmauu_stop_1 = (3 * loop_fac) * logfunc2(m_stop_1sq) \
        * (np.power(yt_wk, 2) - gz_sq
           - (stopuu_num / (m_stop_2sq - m_stop_1sq)))
    sigmauu_stop_2 = (3 * loop_fac) * logfunc2(m_stop_2sq) \
        * (np.power(yt_wk, 2) - gz_sq
           + (stopuu_num / (m_stop_2sq - m_stop_1sq)))
    sigmadd_stop_1 = (3 * loop_fac) * logfunc2(m_stop_1sq) \
        * (gz_sq - (stopdd_num / (m_stop_2sq - m_stop_1sq)))
    sigmadd_stop_2 = (3 * loop_fac) * logfunc2(m_stop_2sq) \
        * (gz_sq + (stopdd_num / (m_stop_2sq - m_stop_1sq)))

    ##### Radiative corrections in sbottom squark sector #####
    delta_sbot = ((1 / 2) - (2 / 3) * sinsq_th_w) * 2 \
        * (((mQ3_sq_wk - mD3_sq_wk) / 2)
           - (mz_q_sq) * cos2b * ((1 / 4) - (1 / 3) * sinsq_th_w))
    sbotuu_num = np.power(ab_wk, 2) - (2 * gz_sq * delta_sbot)
    sbotdd_num = (np.power(yb_wk, 2) * np.power(mu_wk, 2))\
        + (2 * gz_sq * delta_sbot)
    sigmauu_sbot_1 = (3 * loop_fac) * logfunc2(m_sbot_1sq) \
        * (np.power(yb_wk, 2) - gz_sq
           - (sbotuu_num / (m_sbot_2sq - m_sbot_1sq)))
    sigmauu_sbot_2 = (3 * loop_fac) * logfunc2(m_sbot_2sq) \
        * (np.power(yb_wk, 2) - gz_sq
           + (sbotuu_num / (m_sbot_2sq - m_sbot_1sq)))
    sigmadd_sbot_1 = (3 * loop_fac) * logfunc2(m_sbot_1sq) \
        * (gz_sq - (sbotdd_num / (m_sbot_2sq - m_sbot_1sq)))
    sigmadd_sbot_2 = (3 * loop_fac) * logfunc2(m_sbot_2sq) \
        * (gz_sq + (sbotdd_num / (m_sbot_2sq - m_sbot_1sq)))
    
    ##### Radiative corrections in stau slepton sector #####
    delta_stau = ((1 / 2) - (2 * sinsq_th_w)) * 2 \
        * (((mL3_sq_wk - mE3_sq_wk) / 2)
           - (mz_q_sq) * cos2b * ((1 / 4) - sinsq_th_w))
    stauuu_num = np.power(atau_wk, 2) - (2 * gz_sq * delta_stau)
    staudd_num = (np.power(ytau_wk, 2) * np.power(mu_wk, 2))\
        + (2 * gz_sq * delta_stau)
    sigmauu_stau_1 = (loop_fac) * logfunc2(m_stau_1sq) \
        * (np.power(ytau_wk, 2) - gz_sq
           - (stauuu_num / (m_stau_2sq - m_stau_1sq)))
    sigmauu_stau_2 = (loop_fac) * logfunc2(m_stau_2sq) \
        * (np.power(ytau_wk, 2) - gz_sq
           + (stauuu_num / (m_stau_2sq - m_stau_1sq)))
    sigmadd_stau_1 = (loop_fac) * logfunc2(m_stau_1sq) \
        * (gz_sq - (staudd_num / (m_stau_2sq - m_stau_1sq)))
    sigmadd_stau_2 = (loop_fac) * logfunc2(m_stau_2sq) \
        * (gz_sq + (staudd_num / (m_stau_2sq - m_stau_1sq)))
    sigmauu_stau_sneut = ((-4) * loop_fac)\
        * (1 / 2) * gz_sq * logfunc2(mstauneutsq)
    sigmadd_stau_sneut = (4 * loop_fac)\
        * (1 / 2) * gz_sq * logfunc2(mstauneutsq)

    ##### Radiative corrections from 2nd generation sfermions #####
    sigmauu_sstrange_l = ((-12) * loop_fac) * gz_sq \
        * (((-1) / 2) + (1 / 3) * sinsq_th_w) * logfunc2(mQ2_sq_wk)
    sigmauu_sstrange_r = ((-12) * loop_fac) * gz_sq \
        * (((-1) / 3) * sinsq_th_w) * logfunc2(mD2_sq_wk)
    sigmauu_scharm_l = ((-12) * loop_fac) * gz_sq \
        * ((1 / 2) - (2 / 3) * sinsq_th_w) * logfunc2(mQ2_sq_wk)
    sigmauu_scharm_r = ((-12) * loop_fac) * gz_sq \
        * ((2 / 3) * sinsq_th_w) * logfunc2(mU2_sq_wk)
    sigmadd_sstrange_l = (12 * loop_fac) * gz_sq \
        * (((-1) / 2) + (1 / 3) * sinsq_th_w) * logfunc2(mQ2_sq_wk)
    sigmadd_sstrange_r = (12 * loop_fac) * gz_sq \
        * (((-1) / 3) * sinsq_th_w) * logfunc2(mD2_sq_wk)
    sigmadd_scharm_l = (12 * loop_fac) * gz_sq \
        * ((1 / 2) - (2 / 3) * sinsq_th_w) * logfunc2(mQ2_sq_wk)
    sigmadd_scharm_r = (12 * loop_fac) * gz_sq \
        * ((2 / 3) * sinsq_th_w) * logfunc2(mU2_sq_wk)

    ##### Radiative corrections from 2nd generation sleptons/sneutrinos #####
    sigmauu_smu_l = ((-4) * loop_fac) * gz_sq \
        * (((-1) / 2) + sinsq_th_w) * logfunc2(mL2_sq_wk)
    sigmauu_smu_r = ((-4) * loop_fac) * gz_sq \
        * ((-1) * sinsq_th_w) * logfunc2(mE2_sq_wk)
    sigmauu_smu_sneut = ((-4) * loop_fac) * gz_sq \
        * (1 / 2) * logfunc2(msmuneutsq)
    sigmadd_smu_l = (4 * loop_fac) * gz_sq \
        * (((-1) / 2) + sinsq_th_w) * logfunc2(mL2_sq_wk)
    sigmadd_smu_r = (4 * loop_fac) * gz_sq \
        * ((-1) * sinsq_th_w) * logfunc2(mE2_sq_wk)
    sigmadd_smu_sneut = (4 * loop_fac) * gz_sq \
        * (1 / 2) * logfunc2(msmuneutsq)

    ##### Radiative corrections from 1st generation sfermions #####
    sigmauu_sdown_l = ((-12) * loop_fac) * gz_sq \
        * (((-1) / 2) + (1 / 3) * sinsq_th_w) * logfunc2(mQ1_sq_wk)
    sigmauu_sdown_r = ((-12) * loop_fac) * gz_sq \
        * (((-1) / 3) * sinsq_th_w) * logfunc2(mD1_sq_wk)
    sigmauu_sup_l = ((-12) * loop_fac) * gz_sq \
        * ((1 / 2) - (2 / 3) * sinsq_th_w) * logfunc2(mQ1_sq_wk)
    sigmauu_sup_r = ((-12) * loop_fac) * gz_sq \
        * ((2 / 3) * sinsq_th_w) * logfunc2(mU1_sq_wk)
    sigmadd_sdown_l = (12 * loop_fac) * gz_sq \
        * (((-1) / 2) + (1 / 3) * sinsq_th_w) * logfunc2(mQ1_sq_wk)
    sigmadd_sdown_r = (12 * loop_fac) * gz_sq \
        * (((-1) / 3) * sinsq_th_w) * logfunc2(mD1_sq_wk)
    sigmadd_sup_l = (12 * loop_fac) * gz_sq \
        * ((1 / 2) - (2 / 3) * sinsq_th_w) * logfunc2(mQ1_sq_wk)
    sigmadd_sup_r = (12 * loop_fac) * gz_sq \
        * ((2 / 3) * sinsq_th_w) * logfunc2(mU1_sq_wk)

    ##### Radiative corrections from 1st generation sleptons/sneutrinos #####
    sigmauu_selec_l = ((-4) * loop_fac) * gz_sq \
        * (((-1) / 2) + sinsq_th_w) * logfunc2(mL1_sq_wk)
    sigmauu_selec_r = ((-4) * loop_fac) * gz_sq \
        * ((-1) * sinsq_th_w) * logfunc2(mE1_sq_wk)
    sigmauu_selec_sneut = ((-4) * loop_fac) * gz_sq \
        * (1 / 2) * logfunc2(mselecneutsq)
    sigmadd_selec_l = (4 * loop_fac) * gz_sq \
        * (((-1) / 2) + sinsq_th_w) * logfunc2(mL1_sq_wk)
    sigmadd_selec_r = (4 * loop_fac) * gz_sq \
        * ((-1) * sinsq_th_w) * logfunc2(mE1_sq_wk)
    sigmadd_selec_sneut = (4 * loop_fac) * gz_sq \
        * (1 / 2) * logfunc2(mselecneutsq)


    ##### Radiative corrections from neutralino sector #####
    def neutralinouu_deriv_num(msn):
        """
        Return numerator for one-loop uu correction
            derivative term of neutralino.

        Parameters
        ----------
        msn : Float. 
            Neutralino mass used for evaluating results.

        """
        cubicterm = np.power(g2_wk, 2) + np.power(gpr_wk, 2)
        quadrterm = (((np.power(g2_wk, 2) * M2_wk * mu_wk)
                      + (np.power(gpr_wk, 2) * M1_wk * mu_wk))
                     / (np.tan(beta_wk))) \
            - ((np.power((g2_wk * M1_wk), 2)) + (np.power((gpr_wk * M2_wk), 2))
               + ((np.power(g2_wk, 2) + np.power(gpr_wk, 2))
                  * (np.power(mu_wk, 2)))
               + ((np.power((np.power(g2_wk, 2) + np.power(gpr_wk, 2)), 2) / 2)
                  * np.power(vHiggs_wk, 2)))
        linterm = (((-1) * mu_wk) * (((np.power(g2_wk, 2) * M2_wk
                                       * (np.power(M1_wk, 2)
                                          + np.power(mu_wk, 2))))
                                     + (np.power(gpr_wk, 2) * M1_wk
                                        * (np.power(M2_wk, 2)
                                           + np.power(mu_wk, 2))))
                   / np.tan(beta_wk))\
            + ((np.power(((np.power(g2_wk, 2) * M1_wk)
                          + (np.power(gpr_wk, 2) * M2_wk)), 2) / 2)
               * np.power(vHiggs_wk, 2))\
            + (np.power(mu_wk, 2) * ((np.power((g2_wk * M1_wk), 2))
                                     + (np.power((gpr_wk * M2_wk), 2))))\
            + (np.power((np.power(g2_wk, 2) + np.power(gpr_wk, 2)), 2)
               * np.power((vHiggs_wk * mu_wk), 2) * cossqb)
        constterm = (M1_wk * M2_wk * ((np.power(g2_wk, 2) * M1_wk)
                                      + (np.power(gpr_wk, 2) * M2_wk))
                     * np.power(mu_wk, 3) * (1 / np.tan(beta_wk)))\
            - (np.power(((np.power(g2_wk, 2) * M1_wk)
                         + (np.power(gpr_wk, 2) * M2_wk)), 2)
               * np.power(vHiggs_wk, 2) * np.power(mu_wk, 2) * cossqb)
        mynum = (cubicterm * np.power(msn, 6))\
            + (quadrterm * np.power(msn, 4))\
            + (linterm * np.power(msn, 2)) + constterm
        return mynum

    def neutralinodd_deriv_num(msn):
        """
        Return numerator for one-loop dd correction derivative term of
            neutralino.

        Parameters
        ----------
        msn : Float. 
            Neutralino mass used for evaluating results.

        """
        cubicterm = np.power(g2_wk, 2) + np.power(gpr_wk, 2)
        quadrterm = (((np.power(g2_wk, 2) * M2_wk * mu_wk)
                      + (np.power(gpr_wk, 2) * M1_wk * mu_wk))
                     * (np.tan(beta_wk))) \
            - ((np.power((g2_wk * M1_wk), 2)) + (np.power((gpr_wk * M2_wk), 2))
               + ((np.power(g2_wk, 2) + np.power(gpr_wk, 2))
                  * (np.power(mu_wk, 2)))
               + ((np.power((np.power(g2_wk, 2) + np.power(gpr_wk, 2)), 2) / 2)
                  * np.power(vHiggs_wk, 2)))
        linterm = (((-1) * mu_wk)
                   * ((np.power(g2_wk, 2) * M2_wk
                       * (np.power(M1_wk, 2) + np.power(mu_wk, 2)))
                      + (np.power(gpr_wk, 2) * M1_wk
                         * (np.power(M2_wk, 2) + np.power(mu_wk, 2))))
                   * np.tan(beta_wk))\
            + ((np.power(((np.power(g2_wk, 2) * M1_wk)
                          + (np.power(gpr_wk, 2) * M2_wk)), 2) / 2)
               * np.power(vHiggs_wk, 2))\
            + (np.power(mu_wk, 2) * ((np.power((g2_wk * M1_wk), 2))
                                     + (np.power(gpr_wk, 2)
                                        * np.power(M2_wk, 2))))\
            + (np.power((np.power(g2_wk, 2) + np.power(gpr_wk, 2)), 2)
               * np.power((vHiggs_wk * mu_wk), 2) * sinsqb)
        constterm = (M1_wk * M2_wk * ((np.power(g2_wk, 2) * M1_wk)
                                      + (np.power(gpr_wk, 2) * M2_wk))
                     * np.power(mu_wk, 3) * np.tan(beta_wk))\
            - (np.power(((np.power(g2_wk, 2) * M1_wk)
                         + (np.power(gpr_wk, 2) * M2_wk)), 2)
               * np.power((vHiggs_wk * mu_wk), 2) * sinsqb)
        mynum = (cubicterm * np.power(msn, 6)) + (quadrterm * np.power(msn, 4)) \
                + (linterm * np.power(msn, 2)) + constterm
        return mynum

    def neutralino_deriv_denom(msn):
        """
        Return denominator for one-loop correction derivative term of
            neutralino.

        Parameters
        ----------
        msn : Float.
            Neutralino mass.

        """
        quadrterm = (-3) * ((np.power(M1_wk, 2)) + (np.power(M2_wk, 2))
                            + ((np.power(g2_wk, 2) + np.power(gpr_wk, 2))
                               * np.power(vHiggs_wk, 2))
                            + (2 * np.power(mu_wk, 2)))
        linterm = (np.power(vHiggs_wk, 4)
                   * np.power((np.power(g2_wk, 2)
                               + np.power(gpr_wk, 2)), 2) / 2)\
            + (np.power(vHiggs_wk, 2)
               * (2 * ((np.power((g2_wk * M1_wk), 2))
                       + (np.power((gpr_wk * M2_wk), 2))
                       + ((np.power(g2_wk, 2) + np.power(gpr_wk, 2))
                          * np.power(mu_wk, 2))
                       - (mu_wk * ((np.power(gpr_wk, 2) * M1_wk)
                                   + (np.power(g2_wk, 2) * M2_wk))
                          * 2 * np.sin(beta_wk) * np.cos(beta_wk)))))\
            + (2 * ((np.power((M1_wk * M2_wk), 2))
                    + (2 * (np.power((M1_wk * mu_wk), 2)
                            + np.power((M2_wk * mu_wk), 2)))
                    + (np.power(mu_wk, 4))))
        constterm = (np.power(vHiggs_wk, 4) * (1 / 8)
                     * ((np.power((np.power(g2_wk, 2)
                                   + np.power(gpr_wk, 2)), 2)
                         * np.power(mu_wk, 2) * (np.power(cossqb, 2)
                                                 - (6 * cossqb * sinsqb)
                                                 + np.power(sinsqb, 2)))
                        - (2 * np.power(((np.power(g2_wk, 2) * M1_wk)
                                         + (np.power(gpr_wk, 2) * M2_wk)), 2))
                        - (np.power(mu_wk, 2)
                           * np.power((np.power(g2_wk, 2)
                                       + np.power(gpr_wk, 2)), 2))))\
            + (np.power(vHiggs_wk, 2) * 2 * mu_wk
               * ((np.sin(beta_wk) * np.cos(beta_wk))
                  * ((np.power(g2_wk, 2) * M2_wk
                      * (np.power(M1_wk, 2) + np.power(mu_wk, 2)))
                     + (np.power(gpr_wk, 2) * M1_wk
                        * (np.power(M2_wk, 2) + np.power(mu_wk, 2))))))\
            - ((2 * np.power((M2_wk * M1_wk * mu_wk), 2))
               + (np.power(mu_wk, 4) * (np.power(M1_wk, 2)
                                        + np.power(M2_wk, 2))))
        mydenom = (4 * np.power(msn, 6)) + (quadrterm * np.power(msn, 4))\
            + (linterm * np.power(msn, 2)) + constterm
        return mydenom

    def sigmauu_neutralino(msn):
        """
        Return one-loop correction Sigma_u^u(neutralino).

        Parameters
        ----------
        msn : Float.
            Neutralino mass.

        """
        sigma_uu_neutralino = ((-1) * loop_fac) \
            * ((neutralinouu_deriv_num(msn) / neutralino_deriv_denom(msn))
               * logfunc(msn))
        return sigma_uu_neutralino

    def sigmadd_neutralino(msn):
        """
        Return one-loop correction Sigma_d^d(neutralino).

        Parameters
        ----------
        msn : Float.
            Neutralino mass.

        """
        sigma_dd_neutralino = ((-1) * loop_fac)\
            * ((neutralinodd_deriv_num(msn) / neutralino_deriv_denom(msn))
               * logfunc(msn))
        return sigma_dd_neutralino

    ##### Radiative corrections from chargino sector #####
    chargino_numuu = ((-2) * m_w_sq * cos2b) + np.power(M2_wk, 2)\
        + np.power(mu_wk, 2)
    chargino_numdd = (2 * m_w_sq * cos2b) + np.power(M2_wk, 2)\
        + np.power(mu_wk, 2)
    chargino_den = msC2sq - msC1sq
    sigmauu_chargino1 = (-1) * (np.power(g2_wk, 2) * loop_fac)\
        * (1 - (chargino_numuu / chargino_den)) * logfunc2(msC1sq)
    sigmauu_chargino2 = (-1) * (np.power(g2_wk, 2) * loop_fac)\
        * (1 + (chargino_numuu / chargino_den)) * logfunc2(msC2sq)
    sigmadd_chargino1 = (-1) * (np.power(g2_wk, 2) * loop_fac)\
        * (1 - (chargino_numdd / chargino_den)) * logfunc2(msC1sq)
    sigmadd_chargino2 = (-1) * (np.power(g2_wk, 2) * loop_fac)\
        * (1 + (chargino_numdd / chargino_den)) * logfunc2(msC2sq)

    ##### Radiative corrections from Higgs bosons sector #####
    sigmauu_higgs_num = mz_q_sq + (mA0sq * (1 + (4 * cos2b)
                                            + (2 * np.power(cos2b, 2))))
    sigmadd_higgs_num = mz_q_sq + (mA0sq * (1 - (4 * cos2b)
                                            + (2 * np.power(cos2b, 2))))
    higgs_den = mH0sq - mh0sq
    sigmauu_h0 = (gz_sq * loop_fac)\
        * (1 - (sigmauu_higgs_num / higgs_den)) * logfunc2(mh0sq)
    sigmauu_heavy_h0 = (gz_sq * loop_fac)\
        * (1 + (sigmauu_higgs_num / higgs_den)) * logfunc2(mH0sq)
    sigmadd_h0 = (gz_sq * loop_fac)\
        * (1 - (sigmadd_higgs_num / higgs_den)) * logfunc2(mh0sq)
    sigmadd_heavy_h0 = (gz_sq * loop_fac)\
        * (1 + (sigmadd_higgs_num / higgs_den)) * logfunc2(mH0sq)
    sigmauu_h_pm  = (np.power((g2_wk), 2) * loop_fac / 2) * logfunc2(mH_pmsq)
    sigmadd_h_pm = sigmauu_h_pm

    ##### Radiative corrections from weak vector bosons sector #####
    sigmauu_w_pm = (3 * np.power((g2_wk), 2) * loop_fac / 2) * logfunc2(m_w_sq)
    sigmadd_w_pm = sigmauu_w_pm
    sigmauu_z0 = (3 / 4) * loop_fac * (np.power(gpr_wk, 2)
                                       + np.power(g2_wk, 2))\
        * logfunc2(mz_q_sq)
    sigmadd_z0 = sigmauu_z0

    ##### Radiative corrections from SM fermions sector #####
    sigmauu_top = (-1) * np.power(yt_wk, 2) * loop_fac * logfunc(mymt)
    sigmadd_bottom = (-1) * np.power(yb_wk, 2) * loop_fac * logfunc(mymb)
    sigmadd_tau = (-1) * np.power(ytau_wk, 2) * loop_fac * logfunc(mymtau)

    ##### Radiative corrections from two-loop O(alpha_t alpha_s) sector #####
    # Corrections come from Dedes, Slavich paper, arXiv:hep-ph/0212132.
    # alpha_i = y_i^2 / (4 * pi)
    def sigmauu_2loop():
        def Deltafunc(x, y, z):
            mydelta = np.power(x, 2) + np.power(y, 2) + np.power(z, 2)\
                - (2 * ((x * y) + (x * z) + (y * z)))
            return mydelta
    
        def Phifunc(x, y, z):
            if(x / z < 1 and y / z < 1):
                myu = x / z
                myv = y / z
                mylambda = np.sqrt(np.power((1 - myu - myv), 2)
                                   - (4 * myu * myv))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (1 / mylambda) * ((2 * np.log(myxp) * np.log(myxm))
                                          - (np.log(myu) * np.log(myv))
                                          - (2 * (spence(1 - myxp)
                                                  + spence(1 - myxm)))
                                          + (np.power(np.pi, 2) / 3))
            elif(x / z > 1 and y / z < 1):
                myu = z / x
                myv = y / x
                mylambda = np.sqrt(np.power((1 - myu - myv), 2)
                                   - (4 * myu * myv))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / x) * (1 / mylambda) * ((2 * np.log(myxp)
                                                     * np.log(myxm))
                                                    - (np.log(myu)
                                                       * np.log(myv))
                                                    - (2
                                                       * (spence(1 - myxp)
                                                          + spence(1 - myxm)))
                                                    + (np.power(np.pi, 2) / 3))
            elif(x/z > 1 and y/ z > 1 and x > y):
                myu = z / x
                myv = y / x
                mylambda = np.sqrt(np.power((1 - myu - myv), 2)
                                   - (4 * myu * myv))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / x) * (1 / mylambda) * ((2 * np.log(myxp)
                                                     * np.log(myxm))
                                                    - (np.log(myu)
                                                       * np.log(myv))
                                                    - (2 
                                                       * (spence(1 - myxp)
                                                          + spence(1 - myxm)))
                                                    + (np.power(np.pi, 2) / 3))
            elif(x / z < 1 and y / z > 1):
                myu = z / y
                myv = x / y
                mylambda = np.sqrt(np.power((1 - myu - myv), 2)
                                   - (4 * myu * myv))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / y) * (1 / mylambda) * ((2 * np.log(myxp)
                                                     * np.log(myxm))
                                                    - (np.log(myu)
                                                       * np.log(myv))
                                                    - (2
                                                       * (spence(1 - myxp)
                                                          + spence(1 - myxm)))
                                                    + (np.power(np.pi, 2) / 3))
            elif (x / z > 1 and y / z > 1 and y > x):
                myu = z / y
                myv = x / y
                mylambda = np.sqrt(np.power((1 - myu - myv), 2)
                                   - (4 * myu * myv))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / y) * (1 / mylambda) * ((2 * np.log(myxp)
                                                     * np.log(myxm))
                                                    - (np.log(myu)
                                                       * np.log(myv))
                                                    - (2
                                                       * (spence(1 - myxp)
                                                          + spence(1 - myxm)))
                                                    + (np.power(np.pi, 2) / 3))
            return myphi
    
        mst1sq = m_stop_1sq
        mst2sq = m_stop_2sq
        s2theta = 2 * mymt * ((at_wk / yt_wk) + (mu_wk / np.tan(beta_wk)))\
            / (mst1sq - mst2sq)
        s2sqtheta = np.power(s2theta, 2)
        c2sqtheta = 1 - s2sqtheta
        mglsq = np.power(M3_wk, 2)
        myunits = np.power(g3_wk, 2) * 4 * loop_fac_sq
        Q_renorm_sq = np.power(myQ, 2)
        myF = myunits\
            * (((4 * M3_wk * mymt / s2theta) * (1 + (4 * c2sqtheta)))
               - (((2 * (mst1sq - mst2sq)) + (4 * M3_wk * mymt / s2theta))
                  * np.log(mglsq / Q_renorm_sq)
                  * np.log(mymtsq / Q_renorm_sq))
               - (2 * (4 - s2sqtheta) * (mst1sq - mst2sq))
               + ((((4 * mst1sq * mst2sq)
                    - s2sqtheta * np.power((mst1sq + mst2sq), 2))
                   / (mst1sq - mst2sq))
                  * (np.log(mst1sq / Q_renorm_sq))
                  * (np.log(mst2sq / Q_renorm_sq)))
                 + ((((4 * (mglsq + mymtsq + (2 * mst1sq)))
                      - (s2sqtheta * ((3 * mst1sq) + mst2sq))
                      - ((16 * c2sqtheta * M3_wk * mymt * mst1sq)
                         / (s2theta * (mst1sq - mst2sq)))
                      - (4 * s2theta * M3_wk * mymt))
                     * np.log(mst1sq / Q_renorm_sq))
                    + ((mst1sq / (mst1sq - mst2sq))
                       * ((s2sqtheta * (mst1sq + mst2sq))
                          - ((4 * mst1sq) - (2 * mst2sq)))
                       * np.power(np.log(mst1sq / Q_renorm_sq), 2))
                    + (2 * (mst1sq - mglsq - mymtsq
                            + (M3_wk * mymt * s2theta)
                            + ((2 * c2sqtheta * M3_wk * mymt * mst1sq)
                               / (s2theta * (mst1sq - mst2sq))))
                       * np.log(mglsq * mymtsq
                                / (np.power(Q_renorm_sq, 2)))
                       * np.log(mst1sq / Q_renorm_sq))
                    + (((4 * M3_wk * mymt * c2sqtheta * (mymtsq - mglsq))
                        / (s2theta * (mst1sq - mst2sq)))
                       * np.log(mymtsq / mglsq)
                       * np.log(mst1sq / Q_renorm_sq))
                    + (((((4 * mglsq * mymtsq)
                          + (2 * Deltafunc(mglsq, mymtsq, mst1sq))) / mst1sq)
                        - (((2 * M3_wk * mymt * s2theta) / mst1sq)
                           * (mglsq + mymtsq - mst1sq))
                        + ((4 * c2sqtheta * M3_wk * mymt
                            * Deltafunc(mglsq, mymtsq, mst1sq))
                           / (s2theta * mst1sq * (mst1sq - mst2sq))))
                       * Phifunc(mglsq, mymtsq, mst1sq)))
                 - ((((4 * (mglsq + mymtsq + (2 * mst2sq)))
                      - (s2sqtheta * ((3 * mst2sq) + mst1sq))
                      - ((16 * c2sqtheta * M3_wk * mymt * mst2sq)
                         / (((-1) * s2theta) * (mst2sq - mst1sq)))
                      - ((-4) * s2theta * M3_wk * mymt))
                     * np.log(mst2sq / Q_renorm_sq))
                    + ((mst2sq / (mst2sq - mst1sq))
                       * ((s2sqtheta * (mst2sq + mst1sq))
                          - ((4 * mst2sq) - (2 * mst1sq)))
                       * np.power(np.log(mst2sq / Q_renorm_sq), 2))
                    + (2 * (mst2sq - mglsq - mymtsq
                            - (M3_wk * mymt * s2theta)
                            + ((2 * c2sqtheta * M3_wk * mymt * mst2sq)
                               / (s2theta * (mst1sq - mst2sq))))
                       * np.log(mglsq * mymtsq
                                / (np.power(Q_renorm_sq, 2)))
                       * np.log(mst2sq / Q_renorm_sq))
                    + (((4 * M3_wk * mymt * c2sqtheta * (mymtsq - mglsq))
                        / (s2theta * (mst1sq - mst2sq)))
                       * np.log(mymtsq / mglsq)
                       * np.log(mst2sq / Q_renorm_sq))
                    + (((((4 * mglsq * mymtsq)
                          + (2 * Deltafunc(mglsq, mymtsq, mst2sq))) / mst2sq)
                        - ((((-2) * M3_wk * mymt * s2theta) / mst2sq)
                           * (mglsq + mymtsq - mst2sq))
                        + ((4 * c2sqtheta * M3_wk * mymt
                            * Deltafunc(mglsq, mymtsq, mst2sq))
                           / (s2theta * mst2sq * (mst1sq - mst2sq))))
                       * Phifunc(mglsq, mymtsq, mst2sq))))
        myG = myunits\
            * ((5 * M3_wk * s2theta * (mst1sq - mst2sq) / mymt)
               - (10 * (mst1sq + mst2sq - (2 * mymtsq)))
               - (4 * mglsq) + ((12 * mymtsq)
                                * (np.power(np.log(mymtsq / Q_renorm_sq), 2)
                                   - (2 * np.log(mymtsq / Q_renorm_sq))))
               + (((4 * mglsq) - ((M3_wk * s2theta / mymt)
                                  * (mst1sq - mst2sq)))
                  * np.log(mglsq / Q_renorm_sq) * np.log(mymtsq / Q_renorm_sq))
               + (s2sqtheta * (mst1sq + mst2sq)
                  * np.log(mst1sq / Q_renorm_sq)
                  * np.log(mst2sq / Q_renorm_sq))
               + ((((4 * (mglsq + mymtsq + (2 * mst1sq)))
                    + (s2sqtheta * (mst1sq - mst2sq))
                    - ((4 * M3_wk * s2theta / mymt) * (mymtsq + mst1sq)))
                   * np.log(mst1sq / Q_renorm_sq))
                  + (((M3_wk * s2theta * ((5 * mymtsq) - mglsq + mst1sq)
                       / mymt)
                      - (2 * (mglsq + 2 * mymtsq)))
                     * np.log(mymtsq / Q_renorm_sq)
                     * np.log(mst1sq / Q_renorm_sq))
                  + (((M3_wk * s2theta * (mglsq - mymtsq + mst1sq) / mymt)
                      - (2 * mglsq))
                     * np.log(mglsq / Q_renorm_sq)
                     * np.log(mst1sq / Q_renorm_sq))
                  - ((2 + s2sqtheta) * mst1sq
                     * np.power(np.log(mst1sq / Q_renorm_sq), 2))
                  + (((2 * mglsq * (mglsq + mymtsq - mst1sq
                                    - (2 * M3_wk * mymt * s2theta)) / mst1sq)
                      + ((M3_wk * s2theta / (mymt * mst1sq))
                         * Deltafunc(mglsq, mymtsq, mst1sq)))
                     * Phifunc(mglsq, mymtsq, mst1sq)))
               + ((((4 * (mglsq + mymtsq + (2 * mst2sq)))
                    + (s2sqtheta * (mst2sq - mst1sq))
                    - (((-4) * M3_wk * s2theta / mymt) * (mymtsq + mst2sq)))
                   * np.log(mst2sq / Q_renorm_sq))
                  + ((((-1) * M3_wk * s2theta * ((5 * mymtsq) - mglsq + mst2sq)
                       / mymt)
                      - (2 * (mglsq + 2 * mymtsq)))
                     * np.log(mymtsq / Q_renorm_sq)
                     * np.log(mst2sq / Q_renorm_sq))
                  + ((((-1) * M3_wk * s2theta * (mglsq - mymtsq + mst2sq)
                       / mymt)
                      - (2 * mglsq))
                     * np.log(mglsq / Q_renorm_sq)
                     * np.log(mst2sq / Q_renorm_sq))
                  - ((2 + s2sqtheta) * mst2sq
                     * np.power(np.log(mst2sq / Q_renorm_sq), 2))
                  + (((2 * mglsq
                       * (mglsq + mymtsq - mst2sq
                          + (2 * M3_wk * mymt * s2theta)) / mst2sq)
                      + ((M3_wk * (-1) * s2theta / (mymt * mst2sq))
                         * Deltafunc(mglsq, mymtsq, mst2sq)))
                     * Phifunc(mglsq, mymtsq, mst2sq))))
        mysigmauu_2loop = ((mymt * (at_wk / yt_wk) * s2theta * myF)
                           + 2 * np.power(mymt, 2) * myG)\
            / (np.power((vHiggs_wk), 2) * sinsqb)
        return mysigmauu_2loop

    def sigmadd_2loop():
        def Deltafunc(x,y,z):
            mydelta = np.power(x, 2) + np.power(y, 2) + np.power(z, 2)\
                - (2 * ((x * y) + (x * z) + (y * z)))
            return mydelta

        def Phifunc(x,y,z):
            if(x / z < 1 and y / z < 1):
                myu = x / z
                myv = y / z
                mylambda = np.sqrt(np.power((1 - myu - myv), 2)
                                   - (4 * myu * myv))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (1 / mylambda) * ((2 * np.log(myxp) * np.log(myxm))
                                          - (np.log(myu) * np.log(myv))
                                          - (2 * (spence(1 - myxp)
                                                  + spence(1 - myxm)))
                                          + (np.power(np.pi, 2) / 3))
            elif(x / z > 1 and y / z < 1):
                myu = z / x
                myv = y / x
                mylambda = np.sqrt(np.power((1 - myu - myv), 2)
                                   - (4 * myu * myv))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / x) * (1 / mylambda) * ((2 * np.log(myxp)
                                                     * np.log(myxm))
                                                    - (np.log(myu)
                                                       * np.log(myv))
                                                    - (2
                                                       * (spence(1 - myxp)
                                                          + spence(1 - myxm)))
                                                    + (np.power(np.pi, 2) / 3))
            elif(x/z > 1 and y/ z > 1 and x > y):
                myu = z / x
                myv = y / x
                mylambda = np.sqrt(np.power((1 - myu - myv), 2)
                                   - (4 * myu * myv))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / x) * (1 / mylambda) * ((2 * np.log(myxp)
                                                     * np.log(myxm))
                                                    - (np.log(myu)
                                                       * np.log(myv))
                                                    - (2
                                                       * (spence(1 - myxp)
                                                          + spence(1 - myxm)))
                                                    + (np.power(np.pi, 2) / 3))
            elif(x / z < 1 and y / z > 1):
                myu = z / y
                myv = x / y
                mylambda = np.sqrt(np.power((1 - myu - myv), 2)
                                   - (4 * myu * myv))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / y) * (1 / mylambda) * ((2 * np.log(myxp)
                                                     * np.log(myxm))
                                                    - (np.log(myu)
                                                       * np.log(myv))
                                                    - (2
                                                       * (spence(1 - myxp)
                                                          + spence(1 - myxm)))
                                                    + (np.power(np.pi, 2) / 3))
            elif (x / z > 1 and y / z > 1 and y>x):
                myu = z / y
                myv = x / y
                mylambda = np.sqrt(np.power((1 - myu - myv), 2)
                                   - (4 * myu * myv))
                myxp = 0.5 * (1 + myu - myv - mylambda)
                myxm = 0.5 * (1 - myu + myv - mylambda)
                myphi = (z / y) * (1 / mylambda) * ((2 * np.log(myxp)
                                                     * np.log(myxm))
                                                    - (np.log(myu)
                                                       * np.log(myv))
                                                    - (2
                                                       * (spence(1 - myxp)
                                                          + spence(1 - myxm)))
                                                    + (np.power(np.pi, 2) / 3))
            return myphi

        mst1sq = m_stop_1sq
        mst2sq = m_stop_2sq
        Q_renorm_sq=np.power(myQ, 2)
        s2theta = (2 * mymt * ((at_wk / yt_wk) + (mu_wk / np.tan(beta_wk))))\
            / (mst1sq - mst2sq)
        s2sqtheta = np.power(s2theta, 2)
        c2sqtheta = 1 - s2sqtheta
        mglsq = np.power(M3_wk, 2)
        myunits = np.power(g3_wk, 2) * 4\
            / np.power((16 * np.power(np.pi, 2)), 2)
        myF = myunits\
            * ((4 * M3_wk * mymt / s2theta) * (1 + 4 * c2sqtheta)
               - (((2 * (mst1sq - mst2sq))
                  + (4 * M3_wk * mymt / s2theta))
                  * np.log(mglsq / Q_renorm_sq)
                  * np.log(mymtsq / Q_renorm_sq))
               - (2 * (4 - s2sqtheta)
                  * (mst1sq - mst2sq))
               + ((((4 * mst1sq * mst2sq)
                    - s2sqtheta * np.power((mst1sq + mst2sq), 2))
                   / (mst1sq - mst2sq)) * (np.log(mst1sq / Q_renorm_sq))
                  * (np.log(mst2sq / Q_renorm_sq)))
               + ((((4 * (mglsq + mymtsq + (2 * mst1sq)))
                   - (s2sqtheta * ((3 * mst1sq) + mst2sq))
                   - ((16 * c2sqtheta * M3_wk * mymt * mst1sq)
                      / (s2theta * (mst1sq - mst2sq)))
                   - (4 * s2theta * M3_wk * mymt))
                   * np.log(mst1sq / Q_renorm_sq))
                  + ((mst1sq / (mst1sq - mst2sq))
                     * ((s2sqtheta * (mst1sq + mst2sq))
                        - ((4 * mst1sq) - (2 * mst2sq)))
                     * np.power(np.log(mst1sq / Q_renorm_sq), 2))
                  + (2 * (mst1sq - mglsq - mymtsq
                          + (M3_wk * mymt * s2theta)
                          + ((2 * c2sqtheta * M3_wk * mymt * mst1sq)
                             / (s2theta * (mst1sq - mst2sq))))
                     * np.log(mglsq * mymtsq
                              / (np.power(Q_renorm_sq, 2)))
                     * np.log(mst1sq / Q_renorm_sq))
                  + (((4 * M3_wk * mymt * c2sqtheta * (mymtsq - mglsq))
                      / (s2theta * (mst1sq - mst2sq)))
                     * np.log(mymtsq / mglsq)
                     * np.log(mst1sq / Q_renorm_sq))
                  + (((((4 * mglsq * mymtsq)
                        + (2 * Deltafunc(mglsq, mymtsq, mst1sq))) / mst1sq)
                      - (((2 * M3_wk * mymt * s2theta) / mst1sq)
                         * (mglsq + mymtsq - mst1sq))
                      + ((4 * c2sqtheta * M3_wk * mymt
                          * Deltafunc(mglsq, mymtsq, mst1sq))
                         / (s2theta * mst1sq * (mst1sq - mst2sq))))
                     * Phifunc(mglsq, mymtsq, mst1sq)))
               - ((((4 * (mglsq + mymtsq + (2 * mst2sq)))
                   - (s2sqtheta * ((3 * mst2sq) + mst1sq))
                   - ((16 * c2sqtheta * M3_wk * mymt * mst2sq)
                      / (((-1) * s2theta) * (mst2sq - mst1sq)))
                   - ((-4) * s2theta * M3_wk * mymt))
                   * np.log(mst2sq / Q_renorm_sq))
                  + ((mst2sq / (mst2sq - mst1sq))
                     * ((s2sqtheta * (mst2sq + mst1sq))
                        - ((4 * mst2sq) - (2 * mst1sq)))
                     * np.power(np.log(mst2sq / Q_renorm_sq), 2))
                  + (2 * (mst2sq - mglsq - mymtsq
                          - (M3_wk * mymt * s2theta)
                          + ((2 * c2sqtheta * M3_wk * mymt * mst2sq)
                             / (s2theta * (mst1sq - mst2sq))))
                     * np.log(mglsq * mymtsq
                              / (np.power(Q_renorm_sq, 2)))
                     * np.log(mst2sq / Q_renorm_sq))
                  + (((4 * M3_wk * mymt * c2sqtheta * (mymtsq - mglsq))
                      / (s2theta * (mst1sq - mst2sq)))
                     * np.log(mymtsq / mglsq)
                     * np.log(mst2sq / Q_renorm_sq))
                  + (((((4 * mglsq * mymtsq)
                        + (2 * Deltafunc(mglsq, mymtsq, mst2sq))) / mst2sq)
                      - ((((-2) * M3_wk * mymt * s2theta) / mst2sq)
                         * (mglsq + mymtsq - mst2sq))
                      + ((4 * c2sqtheta * M3_wk * mymt
                          * Deltafunc(mglsq, mymtsq, mst2sq))
                         / (s2theta * mst2sq * (mst1sq - mst2sq))))
                     * Phifunc(mglsq, mymtsq, mst2sq))))
        mysigmadd_2loop = (mymt * mu_wk * (1 / np.tan(beta_wk))
                           * s2theta * myF)\
            / (np.power((vHiggs_wk), 2) * cossqb)
        return mysigmadd_2loop

    ##### Total radiative corrections to be used in B*mu evaluation #####
    sigmauu_tot = sigmauu_stop_1 + sigmauu_stop_2 + sigmauu_sbot_1\
        + sigmauu_sbot_2 + sigmauu_stau_1 + sigmauu_stau_2\
        + sigmauu_stau_sneut + sigmauu_scharm_l \
        + sigmauu_scharm_r + sigmauu_sstrange_l + sigmauu_sstrange_r\
        + sigmauu_smu_l + sigmauu_smu_r + sigmauu_smu_sneut + sigmauu_sup_l\
        + sigmauu_sup_r + sigmauu_sdown_l + sigmauu_sdown_r + sigmauu_selec_l\
        + sigmauu_selec_r + sigmauu_selec_sneut + sigmauu_neutralino(msN1)\
        + sigmauu_neutralino(msN2) + sigmauu_neutralino(msN3)\
        + sigmauu_neutralino(msN4) + sigmauu_chargino1 + sigmauu_chargino2\
        + sigmauu_h0 + sigmauu_heavy_h0 + sigmauu_h_pm + sigmauu_w_pm\
        + sigmauu_z0 + sigmauu_top# + sigmauu_2loop()

    sigmadd_tot = sigmadd_stop_1 + sigmadd_stop_2 + sigmadd_sbot_1\
        + sigmadd_sbot_2 + sigmadd_stau_1 + sigmadd_stau_2\
        + sigmadd_stau_sneut + sigmadd_scharm_l \
        + sigmadd_scharm_r + sigmadd_sstrange_l + sigmadd_sstrange_r\
        + sigmadd_smu_l + sigmadd_smu_r + sigmadd_smu_sneut + sigmadd_sup_l\
        + sigmadd_sup_r + sigmadd_sdown_l + sigmadd_sdown_r + sigmadd_selec_l\
        + sigmadd_selec_r + sigmadd_selec_sneut + sigmadd_neutralino(msN1)\
        + sigmadd_neutralino(msN2) + sigmadd_neutralino(msN3)\
        + sigmadd_neutralino(msN4) + sigmadd_chargino1 + sigmadd_chargino2\
        + sigmadd_h0 + sigmadd_heavy_h0 + sigmadd_h_pm + sigmadd_w_pm\
        + sigmadd_z0 + sigmadd_bottom + sigmadd_tau# + sigmadd_2loop()

    return [sigmauu_tot, sigmadd_tot]


##### Approximate mHu^2 evolution from Phys.Rev.D 88, 095013 (2013) #####

def delta_mHu_sq_approx(GUT_scale, low_scale, yt_GUT, at_GUT, mQ3_sq_GUT,
                        mU3_sq_GUT):
    delta_mHu_sq_approximate = (((-3) / (8 * np.power(np.pi, 2)))
                                * ((np.power(yt_GUT, 2)
                                    * (mQ3_sq_GUT + mU3_sq_GUT))
                                   + np.power(at_GUT, 2))
                                * np.log(np.power(GUT_scale, 2)
                                         / np.power(low_scale, 2)))
    return delta_mHu_sq_approximate

##### Extra functions to be used #####

def signed_sqrt(sq_inp):
    """
    Compute the signed square root of a squared input quantity.

    Parameters
    ----------
    sq_inp : Float or array of floats.
        Input to operate on with the signed square root function.

    Returns
    -------
    Float or array of floats.
        Return signed square root of the input array.

    """
    return (np.sign(sq_inp) * np.sqrt(np.abs(sq_inp)))

def find_nearest(array, value):
    """
    Find nearest element of input array to input value.

    Parameters
    ----------
    array : Array of floats or ints.
        Input array the user wants the closest element to "value" within.
    value : Float or int.
        Input value for comparing against elements of "array" to find the
        nearest element.

    Returns
    -------
    Array of ints and floats, [int, float].
        Return an array of the form:
            [index of input array with nearest value to input value,
             nearest value of input array]

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return [idx, array[idx]]

##### Main routine #####
if __name__ == "__main__":
    # Change directory to SoftSUSY(v>=4.1.12) main directory
    os.chdir('/usr/local/softsusy-4.1.12')

    # Prompt user for desired order, n, of numerical precision
    # n is an integer, recommended -6, -5, or -4
    numprec = float(input('Input desired order, n, of '
                          + 'SoftSUSY numerical precision, 10^n,'
                          + ' (recommend -3 to -6): '))

    # Prompt user for NUHM-MSSM inputs. Input format will be:
    # [0: m_0(1), 1: m0(2), 2: m0(3), 3: m_1/2, 4: A_0, 5: sgn(mu), 6: mu(GUT),
    #  7: m_A(pole), 8: tanb(MZ=91.1876 GeV)]
    inplist = np.empty(9)
    promptlist = ['Enter m_0(1) [GeV]: ',
                  'Enter m_0(2) [GeV]: ',
                  'Enter m_0(3) [GeV]: ',
                  'Enter m_1/2 [GeV]: ',
                  'Enter A_0 [GeV]: ',
                  'Enter sgn(mu): ',
                  'Enter mu(weak) [GeV]: ',
                  'Enter m_A(pole) [GeV]: ',
                  'Enter tan(beta)(M_Z): ']
    for i in range(0, len(inplist)):
        ele = float(input(promptlist[i]))
        inplist[i] = ele

    # Set some GUT BC values
    m0_1 = inplist[0]
    m0_2 = inplist[1]
    m0_3 = inplist[2]
    m_hf = inplist[3]
    A0 = inplist[4]
    mu_weak = inplist[6]

    # Create temporary input file, run through SoftSUSY, and obtain GUT scale
    temp = tempfile.NamedTemporaryFile('w+t',
                                       dir='/usr/local/softsusy-4.1.12/tmp')
    try:
        temp.write("Block MODSEL         # Model selection\n"
                   + "    1    1     # sugra\n"
                   + "    6    1             "
                   + "# allow first and second generation Yukawas\n\n"
                   + "Block SMINPUTS               # Standard Model inputs\n"
                   + "    1   1.279440000e+02      # alpha^(-1) SM MSbar(MZ)\n"
                   + "    2   1.166378700e-05      # G_Fermi\n"
                   + "    3   1.184000000e-01      # alpha_s(MZ) SM MSbar\n"
                   + "    4   9.118760000e+01      # MZ(pole)\n"
                   + "    5   4.180000000e+00      # mb(mb) SM MSbar\n "
                   + "    6   1.732000000e+02      # mtop(pole)\n"
                   + "    7   1.777000000e+00      # mtau(pole)\n"
                   + "    8   0.000000000e+00      # mnu3(pole)\n"
                   + "   11   5.109989020e-04      # melectron(pole)\n"
                   + "   12   0.000000000e+00      # mnu1(pole)\n"
                   + "   13   1.056583570e-01      # mmuon(pole)\n"
                   + "   14   0.000000000e+00      # mnu2(pole)\n"
                   + "   21   4.750000000e-03      # md(2 GeV) MS-bar\n"
                   + "   22   2.400000000e-03      # mu(2 GeV) MS-bar\n"
                   + "   23   1.040000000e-01      # ms(2 GeV) MS-bar\n"
                   + "   24   1.270000000e+00      # mc(mc) MS-bar\n"
                   + "Block MINPAR		     # Input parameters\n"
                   + "    2   " + '{:.9e}'.format(inplist[3])
                   + "	     # m12\n"
                   + "    3   " + '{:.9e}'.format(inplist[8])
                   + "      # tanb(MZ)\n"
                   + "    4   " + '{:.9e}'.format(inplist[5])
                   + "      # sgn(mu)\n"
                   + "    5   " + '{:.9e}'.format(inplist[4])
                   + "                # A0\n"
                   + "Block EXTPAR           # GUT scale parameters\n"
                   + "    23  " + '{:.9e}'.format(inplist[6])
                   + "                  # mu\n"
                   + "    26  " + '{:.9e}'.format(inplist[7])
                   + "                 # mA(pole)\n"
                   + "    31  " + '{:.9e}'.format(inplist[0])
                   + "                 # mL1\n"
                   + "    32  " + '{:.9e}'.format(inplist[1])
                   + "                 # mL2\n"
                   + "    33  " + '{:.9e}'.format(inplist[2])
                   + "                 # mL3\n"
                   + "    34  " + '{:.9e}'.format(inplist[0])
                   + "                 # mE1\n"
                   + "    35  " + '{:.9e}'.format(inplist[1])
                   + "                 # mE2\n"
                   + "    36  " + '{:.9e}'.format(inplist[2])
                   + "                 # mE3\n"
                   + "    41  " + '{:.9e}'.format(inplist[0])
                   + "                 # mQ1\n"
                   + "    42  " + '{:.9e}'.format(inplist[1])
                   + "                 # mQ2\n"
                   + "    43  " + '{:.9e}'.format(inplist[2])
                   + "                 # mQ3\n"
                   + "    44  " + '{:.9e}'.format(inplist[0])
                   + "                 # mU1\n"
                   + "    45  " + '{:.9e}'.format(inplist[1])
                   + "                 # mU2\n"
                   + "    46  " + '{:.9e}'.format(inplist[2])
                   + "                 # mU3\n"
                   + "    47  " + '{:.9e}'.format(inplist[0])
                   + "                 # mD1\n"
                   + "    48  " + '{:.9e}'.format(inplist[1])
                   + "                 # mD2\n"
                   + "    49  " + '{:.9e}'.format(inplist[2])
                   + "                 # mD3\n\n"
                   + "Block SOFTSUSY # Optional SOFTSUSY-specific parameters\n"
                   + "    0   0.000000000e+00      # Calculate decays \n"
                   + "    1   " + '{:.9e}'.format(np.power(10, numprec))
                   + "   # Numerical precision: suggested range 10^(-3...-6)\n"
                   + "    2   0	     # Quark mixing parameter: see manual\n"
                   + "    3   0.000000000e+00	     "
                   + "# Additional verbose output?\n"
                   + "    4   1           # SUSY scale=<input>*sqrt(mt1*mt2)\n"
                   + "    5   1          # Include 2-loop RGEs\n"
                   + "    6   1.000000000e-15	     # Numerical precision\n"
                   + "    7	3.000000000e+00	     # Number of loops in Higgs"
                   + " mass computation\n"
                   + "   10	0.000000000e+00	     # Force it to SLHA1 output?\n"
                   + "   11	1.000000000e+19	     # Gravitino mass\n"
                   + "   12   1.000000000e+00	     "
                   + "# Print spectrum even when point disallowed\n"
                   + "   13	0.000000000e+00	     "
                   + "# Set a tachyonic A^0 to zero mass\n"
                   + "   19   1.000000000e+00      "
                   + "# Include 3-loop SUSY RGEs\n"
                   + "   20   31      "
                   + "# Include 2-loop g/Yuk corrections: 31 for all\n"
                   + "   21   0.000000000e+00	      "
                   + "# Calculate uncertainties in Higgs mass prediction")
        temp.seek(0)
        #print(temp.read())
    finally:
        filename = temp.name.removeprefix('/usr/local/softsusy-4.1.12/')
        mySLHA = subprocess.run(['./softpoint.x leshouches < ' + filename],
                                shell=True, text=True,
                                capture_output=True).stdout
        SLHA_for_TeV_BCs = pyslha.readSLHA(mySLHA)
        temp.close()
        #print(mySLHA)


    ##### Constants #####

    loop_fac = 1 / (16 * np.power(np.pi, 2))
    loop_fac_sq = np.power(loop_fac, 2)
    b_1l = [33/5, 1, -3]
    b_2l = [[199/25, 27/5, 88/5], [9/5, 25, 24], [11/5, 9, 14]]
    c_2l = [[26/5, 14/5, 18/5], [6, 6, 2], [4, 4, 0]]
    QGUT = SLHA_for_TeV_BCs.blocks['EXTPAR'][0]
    #numpoints = 1000000

    # Re-run through SoftSUSY to find GUT scale BC's for 41 RGE parameters
    temp1 = tempfile.NamedTemporaryFile('w+t',
                                        dir='/usr/local/softsusy-4.1.12/tmp')
    temp2 = tempfile.NamedTemporaryFile('w+t',
                                        dir='/usr/local/softsusy-4.1.12/tmp')
    try:
        temp1.write("Block MODSEL         # Model selection\n"
                    + "    1    1     # sugra\n"
                    + "    6    1             "
                    + "# allow first and second generation Yukawas\n\n"
                    + "Block SMINPUTS               # Standard Model inputs\n"
                    + "    1   1.279440000e+02      # alpha^(-1) SMMSbar(MZ)\n"
                    + "    2   1.166378700e-05      # G_Fermi\n"
                    + "    3   1.184000000e-01      # alpha_s(MZ) SM MSbar\n"
                    + "    4   9.118760000e+01      # MZ(pole)\n"
                    + "    5   4.180000000e+00      # mb(mb) SM MSbar\n "
                    + "    6   1.732000000e+02      # mtop(pole)\n"
                    + "    7   1.777000000e+00      # mtau(pole)\n"
                    + "    8   0.000000000e+00      # mnu3(pole)\n"
                    + "   11   5.109989020e-04      # melectron(pole)\n"
                    + "   12   0.000000000e+00      # mnu1(pole)\n"
                    + "   13   1.056583570e-01      # mmuon(pole)\n"
                    + "   14   0.000000000e+00      # mnu2(pole)\n"
                    + "   21   4.750000000e-03      # md(2 GeV) MS-bar\n"
                    + "   22   2.400000000e-03      # mu(2 GeV) MS-bar\n"
                    + "   23   1.040000000e-01      # ms(2 GeV) MS-bar\n"
                    + "   24   1.270000000e+00      # mc(mc) MS-bar\n"
                    + "Block MINPAR		     # Input parameters\n"
                    + "    2   " + '{:.9e}'.format(inplist[3])
                    + "	     # m12\n"
                    + "    3   " + '{:.9e}'.format(inplist[8])
                    + "      # tanb(MZ)\n"
                    + "    4   " + '{:.9e}'.format(inplist[5])
                    + "      # sgn(mu)\n"
                    + "    5   " + '{:.9e}'.format(inplist[4])
                    + "                # A0\n"
                    + "Block EXTPAR           # GUT scale parameters\n"
                    + "    23  " + '{:.9e}'.format(inplist[6])
                    + "                  # mu\n"
                    + "    26  " + '{:.9e}'.format(inplist[7])
                    + "                 # mA(pole)\n"
                    + "    31  " + '{:.9e}'.format(inplist[0])
                    + "                 # mL1\n"
                    + "    32  " + '{:.9e}'.format(inplist[1])
                    + "                 # mL2\n"
                    + "    33  " + '{:.9e}'.format(inplist[2])
                    + "                 # mL3\n"
                    + "    34  " + '{:.9e}'.format(inplist[0])
                    + "                 # mE1\n"
                    + "    35  " + '{:.9e}'.format(inplist[1])
                    + "                 # mE2\n"
                    + "    36  " + '{:.9e}'.format(inplist[2])
                    + "                 # mE3\n"
                    + "    41  " + '{:.9e}'.format(inplist[0])
                    + "                 # mQ1\n"
                    + "    42  " + '{:.9e}'.format(inplist[1])
                    + "                 # mQ2\n"
                    + "    43  " + '{:.9e}'.format(inplist[2])
                    + "                 # mQ3\n"
                    + "    44  " + '{:.9e}'.format(inplist[0])
                    + "                 # mU1\n"
                    + "    45  " + '{:.9e}'.format(inplist[1])
                    + "                 # mU2\n"
                    + "    46  " + '{:.9e}'.format(inplist[2])
                    + "                 # mU3\n"
                    + "    47  " + '{:.9e}'.format(inplist[0])
                    + "                 # mD1\n"
                    + "    48  " + '{:.9e}'.format(inplist[1])
                    + "                 # mD2\n"
                    + "    49  " + '{:.9e}'.format(inplist[2])
                    + "                 # mD3\n\n"
                    + "Block SOFTSUSY"
                    + " # Optional SOFTSUSY-specific parameters\n"
                    + "    0   0.000000000e+00      # Calculate decays \n"
                    + "    1   " + '{:.9e}'.format(np.power(10, numprec))
                    + "   # Numerical precision:"
                    + " suggested range 10^(-3...-6)\n"
                    + "    2   0"
                    + "	     # Quark mixing parameter: see manual\n"
                    + "    3   0.000000000e+00	     "
                    + "# Additional verbose output?\n"
                    + "    4   1"
                    + "           # SUSY scale=<input>*sqrt(mt1*mt2)\n"
                    + "    5   1          # Include 2-loop RGEs\n"
                    + "    6   1.000000000e-15	     # Numerical precision\n"
                    + "    7	3.000000000e+00"
                    + "	     # Number of loops in Higgs mass computation\n"
                    + "   10	0.000000000e+00"
                    + "	     # Force it to SLHA1 output?\n"
                    + "   11	1.000000000e+19	     # Gravitino mass\n"
                    + "   12   1.000000000e+00	     "
                    + "# Print spectrum even when point disallowed\n"
                    + "   13	0.000000000e+00	     "
                    + "# Set a tachyonic A^0 to zero mass\n"
                    + "   19   1.000000000e+00      "
                    + "# Include 3-loop SUSY RGEs\n"
                    + "   20   31      "
                    + "# Include 2-loop g/Yuk corrections: 31 for all\n"
                    + "   21   0.000000000e+00	      "
                    + "# Calculate uncertainties in Higgs mass prediction")
        temp2.write("Block MODSEL         # Model selection\n"
                    + "    1    1     # sugra\n"
                    + "    6    1             "
                    + "# allow first and second generation Yukawas\n"
                    + "   12    " + '{:.9e}'.format(QGUT)
                    + " # output at GUT scale\n\n"
                    + "Block SMINPUTS               # Standard Model inputs\n"
                    + "    1   1.279440000e+02      # alpha^(-1) SMMSbar(MZ)\n"
                    + "    2   1.166378700e-05      # G_Fermi\n"
                    + "    3   1.184000000e-01      # alpha_s(MZ) SM MSbar\n"
                    + "    4   9.118760000e+01      # MZ(pole)\n"
                    + "    5   4.180000000e+00      # mb(mb) SM MSbar\n "
                    + "    6   1.732000000e+02      # mtop(pole)\n"
                    + "    7   1.777000000e+00      # mtau(pole)\n"
                    + "    8   0.000000000e+00      # mnu3(pole)\n"
                    + "   11   5.109989020e-04      # melectron(pole)\n"
                    + "   12   0.000000000e+00      # mnu1(pole)\n"
                    + "   13   1.056583570e-01      # mmuon(pole)\n"
                    + "   14   0.000000000e+00      # mnu2(pole)\n"
                    + "   21   4.750000000e-03      # md(2 GeV) MS-bar\n"
                    + "   22   2.400000000e-03      # mu(2 GeV) MS-bar\n"
                    + "   23   1.040000000e-01      # ms(2 GeV) MS-bar\n"
                    + "   24   1.270000000e+00      # mc(mc) MS-bar\n"
                    + "Block MINPAR		     # Input parameters\n"
                    + "    2   " + '{:.9e}'.format(inplist[3])
                    + "	     # m12\n"
                    + "    3   " + '{:.9e}'.format(inplist[8])
                    + "      # tanb(MZ)\n"
                    + "    4   " + '{:.9e}'.format(inplist[5])
                    + "      # sgn(mu)\n"
                    + "    5   " + '{:.9e}'.format(inplist[4])
                    + "                # A0\n"
                    + "Block EXTPAR           # GUT scale parameters\n"
                    + "    23  " + '{:.9e}'.format(inplist[6])
                    + "                  # mu\n"
                    + "    26  " + '{:.9e}'.format(inplist[7])
                    + "                 # mA(pole)\n"
                    + "    31  " + '{:.9e}'.format(inplist[0])
                    + "                 # mL1\n"
                    + "    32  " + '{:.9e}'.format(inplist[1])
                    + "                 # mL2\n"
                    + "    33  " + '{:.9e}'.format(inplist[2])
                    + "                 # mL3\n"
                    + "    34  " + '{:.9e}'.format(inplist[0])
                    + "                 # mE1\n"
                    + "    35  " + '{:.9e}'.format(inplist[1])
                    + "                 # mE2\n"
                    + "    36  " + '{:.9e}'.format(inplist[2])
                    + "                 # mE3\n"
                    + "    41  " + '{:.9e}'.format(inplist[0])
                    + "                 # mQ1\n"
                    + "    42  " + '{:.9e}'.format(inplist[1])
                    + "                 # mQ2\n"
                    + "    43  " + '{:.9e}'.format(inplist[2])
                    + "                 # mQ3\n"
                    + "    44  " + '{:.9e}'.format(inplist[0])
                    + "                 # mU1\n"
                    + "    45  " + '{:.9e}'.format(inplist[1])
                    + "                 # mU2\n"
                    + "    46  " + '{:.9e}'.format(inplist[2])
                    + "                 # mU3\n"
                    + "    47  " + '{:.9e}'.format(inplist[0])
                    + "                 # mD1\n"
                    + "    48  " + '{:.9e}'.format(inplist[1])
                    + "                 # mD2\n"
                    + "    49  " + '{:.9e}'.format(inplist[2])
                    + "                 # mD3\n\n"
                    + "Block SOFTSUSY"
                    + " # Optional SOFTSUSY-specific parameters\n"
                    + "    0   0.000000000e+00      # Calculate decays \n"
                    + "    1   " + '{:.9e}'.format(np.power(10, numprec))
                    + "   # Numerical precision:"
                    + " suggested range 10^(-3...-6)\n"
                    + "    2   0"
                    + "	     # Quark mixing parameter: see manual\n"
                    + "    3   0.000000000e+00	     "
                    + "# Additional verbose output?\n"
                    + "    4   1"
                    + "           # SUSY scale=<input>*sqrt(mt1*mt2)\n"
                    + "    5   1          # Include 2-loop RGEs\n"
                    + "    6   1.000000000e-15	     # Numerical precision\n"
                    + "    7	3.000000000e+00"
                    + "	     # Number of loops in Higgs mass computation\n"
                    + "   10	0.000000000e+00"
                    + "	     # Force it to SLHA1 output?\n"
                    + "   11	1.000000000e+19	     # Gravitino mass\n"
                    + "   12   1.000000000e+00	     "
                    + "# Print spectrum even when point disallowed\n"
                    + "   13	0.000000000e+00	     "
                    + "# Set a tachyonic A^0 to zero mass\n"
                    + "   19   1.000000000e+00      "
                    + "# Include 3-loop SUSY RGEs\n"
                    + "   20   31      "
                    + "# Include 2-loop g/Yuk corrections: 31 for all\n"
                    + "   21   0.000000000e+00	      "
                    + "# Calculate uncertainties in Higgs mass prediction")
        temp1.seek(0)
        temp2.seek(0)
    finally:
        filename1 = temp1.name.removeprefix('/usr/local/softsusy-4.1.12/')
        filename2 = temp2.name.removeprefix('/usr/local/softsusy-4.1.12/')
        myweakSLHA = subprocess.run(['./softpoint.x leshouches < '
                                     + filename1], shell=True, text=True,
                                    capture_output=True).stdout
        myGUTSLHA = subprocess.run(['./softpoint.x leshouches < ' + filename2],
                                   shell=True, text=True,
                                   capture_output=True).stdout
        weak_SLHA = pyslha.readSLHA(myweakSLHA)
        GUT_SLHA = pyslha.readSLHA(myGUTSLHA)
        temp1.close()
        temp2.close()

    # Read off GUT scale BC's
    # GUT_BCs = (0: g1, 1: g2, 2: g3, 3: M1, 4: M2, 5: M3, 6: mu (dummy run),
    #            7: yt, 8: yc,
    #            9: yu, 10: yb, 11: ys, 12: yd, 13: ytau, 14: ymu, 15: ye,
    #            16: at, 17: ac, 18: au, 19: ab, 20: as, 21: ad, 22: atau,
    #            23: amu, 24: ae, 25: mHu^2, 26: mHd^2, 27: mQ1^2,
    #            28: mQ2^2, 29: mQ3^2, 30: mL1^2, 31: mL2^2, 32: mL3^2,
    #            33: mU1^2, 34: mU2^2, 35: mU3^2, 36: mD1^2, 37: mD2^2,
    #            38: mD3^2, 39: mE1^2, 40: mE2^2, 41: mE3^2)
    my_GUT_BCs = [GUT_SLHA.blocks['GAUGE'][1], # g1
                  GUT_SLHA.blocks['GAUGE'][2], # g2
                  GUT_SLHA.blocks['GAUGE'][3], # g3
                  m_hf, # M1
                  m_hf, # M2
                  m_hf, # M3
                  mu_weak, # mu
                  GUT_SLHA.blocks['YU'][3,3], # yt
                  GUT_SLHA.blocks['YU'][2,2], # yc
                  GUT_SLHA.blocks['YU'][1,1], # yu
                  GUT_SLHA.blocks['YD'][3,3], # yb
                  GUT_SLHA.blocks['YD'][2,2], # ys
                  GUT_SLHA.blocks['YD'][1,1], # yd
                  GUT_SLHA.blocks['YE'][3,3], # ytau
                  GUT_SLHA.blocks['YE'][2,2], # ymu
                  GUT_SLHA.blocks['YE'][1,1], # ye
                  GUT_SLHA.blocks['YU'][3,3] * A0, # at
                  GUT_SLHA.blocks['YU'][2,2] * A0, # ac
                  GUT_SLHA.blocks['YU'][1,1] * A0, # au
                  GUT_SLHA.blocks['YD'][3,3] * A0, # ab
                  GUT_SLHA.blocks['YD'][2,2] * A0, # as
                  GUT_SLHA.blocks['YD'][1,1] * A0, # ad
                  GUT_SLHA.blocks['YE'][3,3] * A0, # atau
                  GUT_SLHA.blocks['YE'][2,2] * A0, # amu
                  GUT_SLHA.blocks['YE'][1,1] * A0, # ae
                  #np.power(m0_3, 2), # universal mHu^2
                  #np.power(m0_3, 2), # universal mHd^2
                  GUT_SLHA.blocks['MSOFT'][22], # mHu^2
                  GUT_SLHA.blocks['MSOFT'][21], # mHd^2
                  np.power(m0_1, 2), # mQ1^2
                  np.power(m0_2, 2), # mQ2^2
                  np.power(m0_3, 2), # mQ3^2
                  np.power(m0_1, 2), # mL1^2
                  np.power(m0_2, 2), # mL2^2
                  np.power(m0_3, 2), # mL3^2
                  np.power(m0_1, 2), # mU1^2
                  np.power(m0_2, 2), # mU2^2
                  np.power(m0_3, 2), # mU3^2
                  np.power(m0_1, 2), # mD1^2
                  np.power(m0_2, 2), # mD2^2
                  np.power(m0_3, 2), # mD3^2
                  np.power(m0_1, 2), # mE1^2
                  np.power(m0_2, 2), # mE2^2
                  np.power(m0_3, 2)] # mE3^2
    
    # Read off weak scale values at  MSUSY = sqrt(mt1(MSUSY) * mt2(MSUSY)).
    tanb_weak = weak_SLHA.blocks['HMIX'][2]
    vhiggs_weak = weak_SLHA.blocks['HMIX'][3] / np.sqrt(2)
    #mA_sq_weak = weak_SLHA.blocks['HMIX'][4]
    weak_scale = \
        float(str(weak_SLHA.blocks['GAUGE'])[str(weak_SLHA.blocks['GAUGE']
                                                 ).find('Q=')
                                             +2:str(weak_SLHA.blocks['GAUGE']
                                                    ).find(')')])
    beta_weak = np.arctan(tanb_weak)
    sin2b_weak = np.sin(2 * beta_weak)

    # Evolve all 41 parameters from GUT scale down to obtained weak scale.
    # About 1000 points per decade in solution
    numpoints = int((np.log10(QGUT / weak_scale)) * 1000)
    # Set up array of values for renormalization scale, log. unif. spacing
    t_vals = np.logspace(np.log10(weak_scale), np.log10(QGUT), numpoints)
    my_sols = my_RGE_solver(my_GUT_BCs, QGUT, weak_scale)

    # Set up solution arrays in ascending order of scale, from 1 TeV to GUT
    g1_my_sol = np.array(my_sols[0][::-1])
    g2_my_sol = np.array(my_sols[1][::-1])
    g3_my_sol = np.array(my_sols[2][::-1])
    M1_my_sol = np.array(my_sols[3][::-1])
    M2_my_sol = np.array(my_sols[4][::-1])
    M3_my_sol = np.array(my_sols[5][::-1])
    yt_my_sol = np.array(my_sols[7][::-1])
    yc_my_sol = np.array(my_sols[8][::-1])
    yu_my_sol = np.array(my_sols[9][::-1])
    yb_my_sol = np.array(my_sols[10][::-1])
    ys_my_sol = np.array(my_sols[11][::-1])
    yd_my_sol = np.array(my_sols[12][::-1])
    ytau_my_sol = np.array(my_sols[13][::-1])
    ymu_my_sol = np.array(my_sols[14][::-1])
    ye_my_sol = np.array(my_sols[15][::-1])
    at_my_sol = np.array(my_sols[16][::-1])
    ac_my_sol = np.array(my_sols[17][::-1])
    au_my_sol = np.array(my_sols[18][::-1])
    ab_my_sol = np.array(my_sols[19][::-1])
    as_my_sol = np.array(my_sols[20][::-1])
    ad_my_sol = np.array(my_sols[21][::-1])
    atau_my_sol = np.array(my_sols[22][::-1])
    amu_my_sol = np.array(my_sols[23][::-1])
    ae_my_sol = np.array(my_sols[24][::-1])
    mHu_sq_my_sol = np.array(my_sols[25][::-1])
    mHd_sq_my_sol = np.array(my_sols[26][::-1])
    mQ1_sq_my_sol = np.array(my_sols[27][::-1])
    mQ2_sq_my_sol = np.array(my_sols[28][::-1])
    mQ3_sq_my_sol = np.array(my_sols[29][::-1])
    mL1_sq_my_sol = np.array(my_sols[30][::-1])
    mL2_sq_my_sol = np.array(my_sols[31][::-1])
    mL3_sq_my_sol = np.array(my_sols[32][::-1])
    mU1_sq_my_sol = np.array(my_sols[33][::-1])
    mU2_sq_my_sol = np.array(my_sols[34][::-1])
    mU3_sq_my_sol = np.array(my_sols[35][::-1])
    mD1_sq_my_sol = np.array(my_sols[36][::-1])
    mD2_sq_my_sol = np.array(my_sols[37][::-1])
    mD3_sq_my_sol = np.array(my_sols[38][::-1])
    mE1_sq_my_sol = np.array(my_sols[39][::-1])
    mE2_sq_my_sol = np.array(my_sols[40][::-1])
    mE3_sq_my_sol = np.array(my_sols[41][::-1])

    ##### Compute radiative corrections to Higgs scalar pot'l min. cond's #####
    [my_sigma_uu,
     my_sigma_dd] = my_radcorr_calc(weak_scale, vhiggs_weak, mu_weak,
                                    beta_weak, yt_my_sol[0], yb_my_sol[0],
                                    ytau_my_sol[0], g1_my_sol[0], g2_my_sol[0],
                                    g3_my_sol[0], mQ3_sq_my_sol[0],
                                    mQ2_sq_my_sol[0], mQ1_sq_my_sol[0],
                                    mL3_sq_my_sol[0], mL2_sq_my_sol[0],
                                    mL1_sq_my_sol[0], mU3_sq_my_sol[0],
                                    mU2_sq_my_sol[0], mU1_sq_my_sol[0],
                                    mD3_sq_my_sol[0], mD2_sq_my_sol[0],
                                    mD1_sq_my_sol[0], mE3_sq_my_sol[0],
                                    mE2_sq_my_sol[0], mE1_sq_my_sol[0],
                                    M1_my_sol[0], M2_my_sol[0], M3_my_sol[0],
                                    mHu_sq_my_sol[0], mHd_sq_my_sol[0],
                                    at_my_sol[0], ab_my_sol[0], atau_my_sol[0])
    

    ##### Compute B*mu(weak) from ~two-loop Higgs minimization condition #####
    b_weak = (sin2b_weak / 2.0) * (mHu_sq_my_sol[0] + my_sigma_uu
                                   + mHd_sq_my_sol[0] + my_sigma_dd
                                   + (2 * np.power(mu_weak, 2)))

    ##### Perform second running of RGEs back up to GUT scale, include b #####
    # Read off weak scale BC's
    # weakBCs = (0: g1, 1: g2, 2: g3, 3: M1, 4: M2, 5: M3, 6: mu (full run),
    #            7: yt, 8: yc,
    #            9: yu, 10: yb, 11: ys, 12: yd, 13: ytau, 14: ymu, 15: ye,
    #            16: at, 17: ac, 18: au, 19: ab, 20: as, 21: ad, 22: atau,
    #            23: amu, 24: ae, 25: mHu^2, 26: mHd^2, 27: mQ1^2,
    #            28: mQ2^2, 29: mQ3^2, 30: mL1^2, 31: mL2^2, 32: mL3^2,
    #            33: mU1^2, 34: mU2^2, 35: mU3^2, 36: mD1^2, 37: mD2^2,
    #            38: mD3^2, 39: mE1^2, 40: mE2^2, 41: mE3^2, 42: b)
    my_weak_BCs = [g1_my_sol[0], # g1
                   g2_my_sol[0], # g2
                   g3_my_sol[0], # g3
                   M1_my_sol[0], # M1
                   M2_my_sol[0], # M2
                   M3_my_sol[0], # M3
                   mu_weak, # mu
                   yt_my_sol[0], # yt
                   yc_my_sol[0], # yc
                   yu_my_sol[0], # yu
                   yb_my_sol[0], # yb
                   ys_my_sol[0], # ys
                   yd_my_sol[0], # yd
                   ytau_my_sol[0], # ytau
                   ymu_my_sol[0], # ymu
                   ye_my_sol[0], # ye
                   at_my_sol[0], # at
                   ac_my_sol[0], # ac
                   au_my_sol[0], # au
                   ab_my_sol[0], # ab
                   as_my_sol[0], # as
                   ad_my_sol[0], # ad
                   atau_my_sol[0], # atau
                   amu_my_sol[0], # amu
                   ae_my_sol[0], # ae
                   mHu_sq_my_sol[0], # mHu^2
                   mHd_sq_my_sol[0], # mHd^2
                   mQ1_sq_my_sol[0], # mQ1^2
                   mQ2_sq_my_sol[0], # mQ2^2
                   mQ3_sq_my_sol[0], # mQ3^2
                   mL1_sq_my_sol[0], # mL1^2
                   mL2_sq_my_sol[0], # mL2^2
                   mL3_sq_my_sol[0], # mL3^2
                   mU1_sq_my_sol[0], # mU1^2
                   mU2_sq_my_sol[0], # mU2^2
                   mU3_sq_my_sol[0], # mU3^2
                   mD1_sq_my_sol[0], # mD1^2
                   mD2_sq_my_sol[0], # mD2^2
                   mD3_sq_my_sol[0], # mD3^2
                   mE1_sq_my_sol[0], # mE1^2
                   mE2_sq_my_sol[0], # mE2^2
                   mE3_sq_my_sol[0], # mE3^2
                   b_weak] # b

    ##### Perform second RGE run back up to GUT scale #####
    my_sols_2 = my_RGE_solver_2(my_weak_BCs, QGUT, weak_scale)
    mu_my_sol = np.array(my_sols_2[6])
    b_my_sol = np.array(my_sols_2[42])

    # Set up approximate vs exact arrays

    approx_mHu_sq_radcorr = delta_mHu_sq_approx(QGUT,
                                                t_vals[0:-1],
                                                yt_my_sol[-1],
                                                at_my_sol[-1],
                                                mQ3_sq_my_sol[-1],
                                                mU3_sq_my_sol[-1])
    approx_mHu_sq = (mHu_sq_my_sol[-1]) + approx_mHu_sq_radcorr
    approx_mHu = signed_sqrt(approx_mHu_sq)

    exact_mHu_sq_radcorr = mHu_sq_my_sol[0:-1] - mHu_sq_my_sol[-1]
    exact_mHu = signed_sqrt(mHu_sq_my_sol[:-1])

    # Plot deviation of exact mHu^2 evolution over approximate mHu^2 evolution from Q_GUT to M_SUSY

    # BM point (log x-axis) plot
    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(t_vals[:-1], exact_mHu_sq_radcorr / approx_mHu_sq_radcorr)
    plt.xscale('log')
    plt.xlabel(r'Renormalization Scale $Q$ [GeV]')
    plt.ylabel(r'$\delta m_{H_{u}}^{2}(exact)~/~\delta m_{H_{u}}^{2}(approx.)$'
               )
    plt.title(r'$\delta m_{H_{u}}^{2}$ Approximation Validity in BM Point')
    plt.show()

    # Explicit comparisons of exact and approximate m_Hu in BM point
    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(t_vals[:-1], approx_mHu, linestyle='dashed', color='red')
    plt.plot(t_vals[:-1], exact_mHu, color='black')
    plt.xscale('log')
    plt.xlabel(r'Renormalization Scale $Q$ [GeV]')
    plt.ylabel(r'$m_{H_{u}}(Q)=$'+'sign'
               +r'$(m_{H_{u}}^{2}(Q))\sqrt{|m_{H_{u}}^{2}(Q)|}$ [GeV]')
    plt.title(r'Exact (2-loop) vs. approximate $m_{H_{u}}$ evolution'
              + r' in BM Point')
    plt.legend([r'Approximate $m_{H_{u}}$', r'Exact 2-loop $m_{H_{u}}$'])
    plt.show()

    # Plot mass spectrum
    fig = plt.figure(figsize=(8, 6), dpi=300)
    # plot mHu
    plt.plot(t_vals, signed_sqrt(mHu_sq_my_sol),
             label=r'$m_{H_{u}}$', color='black')
    # plot mHd
    plt.plot(t_vals, signed_sqrt(mHd_sq_my_sol),
             label=r'$m_{H_{d}}$', linestyle='dashed', color='black')
    # plot mQ3
    plt.plot(t_vals, signed_sqrt(mQ3_sq_my_sol),
             label=r'$m_{Q_{3}}$', linestyle='dotted', color='blue')
    # plot mQ2
    plt.plot(t_vals, signed_sqrt(mQ2_sq_my_sol),
             label=r'$m_{Q_{2}}$', color='blue')
    # plot mQ1
    plt.plot(t_vals, signed_sqrt(mQ1_sq_my_sol),
             label=r'$m_{Q_{1}}$', linestyle='dashed', color='blue')
    # plot mL3
    plt.plot(t_vals, signed_sqrt(mL3_sq_my_sol),
             label=r'$m_{L_{3}}$', linestyle='dotted', color='red')
    # plot mL2
    plt.plot(t_vals, signed_sqrt(mL2_sq_my_sol),
             label=r'$m_{L_{2}}$', color='red')
    # plot mL1
    plt.plot(t_vals, signed_sqrt(mL1_sq_my_sol),
             label=r'$m_{L_{1}}$', linestyle='dashed', color='red')
    # plot mU3
    plt.plot(t_vals, signed_sqrt(mU3_sq_my_sol),
             label=r'$m_{U_{3}}$', linestyle='dotted', color='limegreen')
    # plot mU2
    plt.plot(t_vals, signed_sqrt(mU2_sq_my_sol),
             label=r'$m_{U_{2}}$', color='limegreen')
    # plot mU1
    plt.plot(t_vals, signed_sqrt(mU1_sq_my_sol),
             label=r'$m_{U_{1}}$', linestyle='dashed', color='limegreen')
    # plot mD3
    plt.plot(t_vals, signed_sqrt(mD3_sq_my_sol),
             label=r'$m_{D_{3}}$', linestyle='dotted', color='orange')
    # plot mD2
    plt.plot(t_vals, signed_sqrt(mD2_sq_my_sol),
             label=r'$m_{D_{2}}$', color='orange')
    # plot mD1
    plt.plot(t_vals, signed_sqrt(mD1_sq_my_sol),
             label=r'$m_{D_{1}}$', linestyle='dashed', color='orange')
    # plot mE3
    plt.plot(t_vals, signed_sqrt(mE3_sq_my_sol),
             label=r'$m_{E_{3}}$', linestyle='dotted', color='darkviolet')
    # plot mE2
    plt.plot(t_vals, signed_sqrt(mE2_sq_my_sol),
             label=r'$m_{E_{2}}$', color='darkviolet')
    # plot mE1
    plt.plot(t_vals, signed_sqrt(mE1_sq_my_sol),
             label=r'$m_{E_{1}}$', linestyle='dashed', color='darkviolet')
    # plot M3
    plt.plot(t_vals, M3_my_sol,
             label=r'$M_{3}$', linestyle='dotted', color='darkgoldenrod')
    # plot M2
    plt.plot(t_vals, M2_my_sol,
             label=r'$M_{2}$', color='darkgoldenrod')
    # plot M1
    plt.plot(t_vals, M1_my_sol,
             label=r'$M_{1}$', linestyle='dashed', color='darkgoldenrod')
    plt.title('Evolution of soft scalar masses')
    plt.xscale('log')
    plt.legend(loc = 'center left', bbox_to_anchor=(1., 0.5))
    plt.show()

    # plot mu evolutions
    fig = plt.figure(figsize=(8, 6), dpi=300)
    # plot mu(GUT) / mu(weak)
    plt.plot(t_vals, mu_my_sol[-1] / mu_my_sol)
    plt.title(r'Deviation of $\mu(Q)$ from $\mu(GUT)$')
    plt.xscale('log')
    plt.xlabel(r'Renormalization scale $Q$ [GeV]')
    plt.ylabel(r'$\mu(GUT) / \mu(Q)$')
    plt.show()

    # plot mU3^2 evolutions
    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(t_vals, mU3_sq_my_sol / mU3_sq_my_sol[-1])
    plt.title(r'Deviation of $m_{U_{3}}^{2}(Q)$ from $m_{U_{3}}^{2}(GUT)$')
    plt.xscale('log')
    plt.xlabel(r'Renormalization scale $Q$ [GeV]')
    plt.ylabel(r'$m_{U_{3}}^{2}(Q) / m_{U_{3}}^{2}(GUT)$')
    plt.show()

    # plot signed_sqrt(b)=signed_sqrt(B*mu) evolutions
    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(t_vals, signed_sqrt(b_my_sol))
    plt.title(r'Higgs soft bilinear coupling signed $\sqrt{b}=\sqrt{B\mu}$ evolution')
    plt.xscale('log')
    plt.xlabel(r'Renormalization scale $Q$ [GeV]')
    plt.ylabel(r'Signed $\sqrt{b(Q)}$')
    plt.show()

    # plot b=B*mu evolutions
    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(t_vals, b_my_sol)
    plt.title(r'Higgs soft bilinear coupling $b=B\mu$ evolution')
    plt.xscale('log')
    plt.xlabel(r'Renormalization scale $Q$ [GeV]')
    plt.ylabel(r'$b(Q)$')
    plt.show()

    # plot signed_sqrt(B) evolutions
    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(t_vals, signed_sqrt(b_my_sol / mu_my_sol))
    plt.title(r'Higgs soft bilinear coupling signed $\sqrt{B}$ evolution')
    plt.xscale('log')
    plt.xlabel(r'Renormalization scale $Q$ [GeV]')
    plt.ylabel(r'Signed $\sqrt{b(Q)}$')
    plt.show()

    # plot B evolutions
    fig = plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(t_vals, b_my_sol / mu_my_sol)
    plt.title(r'Higgs soft bilinear coupling $B$ evolution')
    plt.xscale('log')
    plt.xlabel(r'Renormalization scale $Q$ [GeV]')
    plt.ylabel(r'$b(Q)$')
    plt.show()
