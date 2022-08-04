#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:21:55 2022

@author: chanceronemus
"""
import uncertainties as unc
import uncertainties.unumpy as unp  
import uncertainties.umath as umath
import numpy as np
import matplotlib.pyplot as plt

### Inputs
### Grove and Harrison, 1996 data

# Temperature Celcius
TC = np.array([550, 550, 600, 600, 650, 650, 700, 700, 700, 700, 700, 700, 
               700, 650, 650, 650, 650, 600, 600, 550, 550, 550, 550, 550])

# Temperature error, assumed to be +/- 5C by Grove and Harrison (1996)
TC_err = np.full(len(TC), float(5))

# Time in days
t_day = np.array([97.13, 97.13, 50.29, 50.27, 28.91, 28.95, 15.89, 15.93, 
                  78.09, 15.07, 22.01, 15.07, 14.03, 25.88, 44.16, 20.98, 
                  14.86, 23.91, 75.93, 78.09, 75.93, 75.93, 67.92, 67.92])

# Radius in microns
a_um = np.array([96, 113, 113, 128, 128, 128, 128, 128, 154, 125, 109, 92, 
                 92, 125, 125, 92, 92, 56, 56, 109, 92, 56, 56, 56])


# Radius uncertainty in microns
a_um_err = np.array([10, 13, 13, 13, 13, 13, 13, 13, 17, 14, 15, 13, 13, 14, 
                     14, 13, 13, 10, 10, 15, 13, 10, 10, 10])

# Fractional loss (f)
f_release = np.array([0.102, 0.074, 0.159, 0.111, 0.181, 0.18, 0.241, 0.233, 
                      0.584, 0.307, 0.25, 0.349, 0.302, 0.191, 0.216, 0.185, 
                      0.121, 0.216, 0.29, 0.0443, 0.0948, 0.117, 0.106, 0.135])

# f uncertainty
f_release_err = np.array([0.006, 0.004, 0.004, 0.004, 0.005, 0.004, 0.003, 
                          0.006, 0.003, 0.006, 0.006, 0.006, 0.006, 0.006, 
                          0.006, 0.004, 0.007, 0.007, 0.007, 0.0101, 0.0076, 
                          0.009, 0.007, 0.007])

### Ideal gas constant (kcal/(K*mol))
R = 0.0019858775

#%%

### Functions for calculating ln(D) for various geometries

# Calculate infinite cylinder ln(D) using Eq. 2 from Grove and Harrison (1996
def calc_inf_cyl_Dta2(f):  
    return (4/np.pi)*((1-umath.sqrt(1-((np.pi/4)*f)))**2)

# Calcuate infinite plane ln(D) using equation for plane from Reiners et al. (2017), Table 5.1
def calc_plane_Dta2(f):
    return ((f*np.sqrt(np.pi)/2)**2)

# Calculates sphere ln(D) using equation for sphere from Reiners et al. (2017), Table 5.1
def calc_sphere_Dta2(f):
    return (1/3)*(-f-(2*(np.sqrt(3)*umath.sqrt(3-f)))+6)
    
# Calcuates ln(D) from Dt/a^2
def calc_lnD(Dta2, a, t):
    return umath.log((Dta2*(a**2))/t)

#%%

### Function for bivariate regression from York (2004):
### https://gist.github.com/mikkopitkanen/ce9cd22645a9e93b6ca48ba32a3c85d0 

def bivariate_fit(xi, yi, dxi, dyi, ri=0.0, b0=1.0, maxIter=1e6):
    """Make a linear bivariate fit to xi, yi data using York et al. (2004).
    This is an implementation of the line fitting algorithm presented in:
    York, D et al., Unified equations for the slope, intercept, and standard
    errors of the best straight line, American Journal of Physics, 2004, 72,
    3, 367-375, doi = 10.1119/1.1632486
    See especially Section III and Table I. The enumerated steps below are
    citations to Section III
    Parameters:
      xi, yi      x and y data points
      dxi, dyi    errors for the data points xi, yi
      ri          correlation coefficient for the weights
      b0          initial guess b
      maxIter     float, maximum allowed number of iterations
    Returns:
      a           y-intercept, y = a + bx
      b           slope
      S           goodness-of-fit estimate
      sigma_a     standard error of a
      sigma_b     standard error of b
    Usage:
    [a, b] = bivariate_fit( xi, yi, dxi, dyi, ri, b0, maxIter)
    """
    # (1) Choose an approximate initial value of b
    b = b0

    # (2) Determine the weights wxi, wyi, for each point.
    wxi = 1.0 / dxi**2.0
    wyi = 1.0 / dyi**2.0

    alphai = (wxi * wyi)**0.5
    b_diff = 999.0

    # tolerance for the fit, when b changes by less than tol for two
    # consecutive iterations, fit is considered found
    tol = 1.0e-8

    # iterate until b changes less than tol
    iIter = 1
    while (abs(b_diff) >= tol) & (iIter <= maxIter):

        b_prev = b

        # (3) Use these weights wxi, wyi to evaluate Wi for each point.
        Wi = (wxi * wyi) / (wxi + b**2.0 * wyi - 2.0*b*ri*alphai)

        # (4) Use the observed points (xi ,yi) and Wi to calculate x_bar and
        # y_bar, from which Ui and Vi , and hence betai can be evaluated for
        # each point
        x_bar = np.sum(Wi * xi) / np.sum(Wi)
        y_bar = np.sum(Wi * yi) / np.sum(Wi)

        Ui = xi - x_bar
        Vi = yi - y_bar

        betai = Wi * (Ui / wyi + b*Vi / wxi - (b*Ui + Vi) * ri / alphai)

        # (5) Use Wi, Ui, Vi, and betai to calculate an improved estimate of b
        b = np.sum(Wi * betai * Vi) / np.sum(Wi * betai * Ui)

        # (6) Use the new b and repeat steps (3), (4), and (5) until successive
        # estimates of b agree within some desired tolerance tol
        b_diff = b - b_prev

        iIter += 1

    # (7) From this final value of b, together with the final x_bar and y_bar,
    # calculate a from
    a = y_bar - b * x_bar

    # Goodness of fit
    S = np.sum(Wi * (yi - b*xi - a)**2.0)

    # (8) For each point (xi, yi), calculate the adjusted values xi_adj
    xi_adj = x_bar + betai

    # (9) Use xi_adj, together with Wi, to calculate xi_adj_bar and thence ui
    xi_adj_bar = np.sum(Wi * xi_adj) / np.sum(Wi)
    ui = xi_adj - xi_adj_bar

    # (10) From Wi , xi_adj_bar and ui, calculate sigma_b, and then sigma_a
    # (the standard uncertainties of the fitted parameters)
    sigma_b = np.sqrt(1.0 / np.sum(Wi * ui**2))
    sigma_a = np.sqrt(1.0 / np.sum(Wi) + xi_adj_bar**2 * sigma_b**2)

    # calculate covariance matrix of b and a (York et al., Section II)
    cov = -xi_adj_bar * sigma_b**2
    cov_matrix = np.array(
        [[sigma_b**2, cov], [cov, sigma_a**2]])

    if iIter <= maxIter:
        return a, b, S, cov_matrix, sigma_a, sigma_b
    else:
        print("bivariate_fit.py exceeded maximum number of iterations, " +
              "maxIter = {:}".format(maxIter))
        return np.nan, np.nan, np.nan, np.nan
    
#%%

### Function to generate Arrhenius plots and calculate diffusion parameters
### using OLS and York (2004) bivariate fit regression

def Arrplot(Sx, Swx, Sy, Swy, Cx, Cwx, Cy, Cwy):
    
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    
    # SPHERE
    # Plot error bars
    ax1.errorbar(Sx, Sy, xerr = Swx, yerr = Swy, fmt = 'o')
    
    # Run bivariate regression
    a_bivar, b_bivar, S, cov, sigma_a, sigma_b = bivariate_fit(
        Sx, Sy, Swx, Swy, b0=0.0)
    
    # Collect slope and intercept for uncertainty propagation
    Sa_unc = unc.ufloat(a_bivar, sigma_a)
    Sb_unc = unc.ufloat(b_bivar, sigma_b)
    
    # Calculate Ea and uncertainty
    SEa_unc = Sb_unc * (-1) * R * 1000
    
    # Plot regression
    label_bivar = 'ln(D)={:1.3f}(\u00B1 {:1.2f})(1000/T)\n+ {:1.2f}(\u00B1 {:1.2f})'.format(b_bivar, sigma_b, a_bivar, sigma_a)
    
    xlim = np.array([np.min(Sx)-0.05, np.max(Sx)+0.05])
    ylim = np.array([np.min(Sy)-1, np.max(Sy)+1])
    
    plt.plot(xlim, b_bivar*xlim + a_bivar,  'b-',
                        label=label_bivar)
    
    # CYLINDER
    # Plot error bars
    ax1.errorbar(Cx, Cy, xerr = Cwx, yerr = Cwy, fmt = 'o')
    
    # Run bivariate regression
    a_bivar, b_bivar, S, cov, sigma_a, sigma_b = bivariate_fit(
        Cx, Cy, Cwx, Cwy, b0=0.0)
    
    # Collect slope and intercept for uncertainty propagation
    Ca_unc = unc.ufloat(a_bivar, sigma_a)
    Cb_unc = unc.ufloat(b_bivar, sigma_b)
    
    # Calculate Ea and uncertainty
    CEa_unc = Cb_unc * (-1) * R * 1000
    
    # Plot regression
    label_bivar = 'ln(D)={:1.3f}(\u00B1 {:1.2f})(1000/T)\n+ {:1.2f}(\u00B1 {:1.2f})'.format(b_bivar, sigma_b, a_bivar, sigma_a)
    
    xlim = np.array([np.min(Sx)-0.05, np.max(Sx)+0.05])
    ylim = np.array([np.min(Sy)-1, np.max(Cy)+1])
    
    plt.plot(xlim, b_bivar*xlim + a_bivar,  'b-',
                        label=label_bivar)
    
    ax1.legend()
        
    ax1.set_xlabel('1000/T (K)')
    ax1.set_ylabel('lnD (cm\u00b2/s)')
    
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    
    ax1.grid(visible=True)
    plt.show()
    
    figname = 'FS1_arrhenius.pdf'
    fig1.savefig(figname)
    
    return Sa_unc, SEa_unc, Ca_unc, CEa_unc

#%%

# Create arrays containing the measured values and uncertainties
a_um_unc = unp.uarray(a_um, a_um_err)
f_release_unc = unp.uarray(f_release, f_release_err)
TC_unc = unp.uarray(TC, TC_err)

# Convert time, temp, and radius to Kelvin, seconds, and cm respectively
TK_unc = TC_unc + 273.15
a_cm_unc = a_um_unc/10000
t_sec = t_day * 86400

# Convert temperature and uncertainty to 1000/T
T_1000_unc = 1000/TK_unc

### Calculate ln(D) for various geometries using equations from Grove and Harrison (1996) and Reiners et al. (2017)
# Infinite cylinder
cyl_Dta2 = [calc_inf_cyl_Dta2(f_release_unc[i]) for i in range(0, len(f_release_unc))]
cyl_lnD = [calc_lnD(cyl_Dta2[i], a_cm_unc[i], t_sec[i]) for i in range(0, len(cyl_Dta2))]

# Infinite plane/slab
plane_Dta2 = np.array([calc_plane_Dta2(i) for i in f_release_unc])
plane_lnD = np.array([calc_lnD(plane_Dta2[i], a_cm_unc[i], t_sec[i]) for i in range(0, len(plane_Dta2))])

# Sphere
sphere_Dta2 = np.array([calc_sphere_Dta2(i) for i in f_release_unc])
sphere_lnD = np.array([calc_lnD(sphere_Dta2[i], a_cm_unc[i], t_sec[i]) for i in range(0, len(sphere_Dta2))])

### Get values and uncertainties into a the right format for regression
T_1000_vals = np.array([unc.nominal_value(T_1000_unc[i]) for i in range(0, len(T_1000_unc))])
T_1000_stds = np.array([unc.std_dev(T_1000_unc[i]) for i in range(0, len(T_1000_unc))])

cyl_lnD_vals = np.array([unc.nominal_value(cyl_lnD[i]) for i in range(0, len(cyl_lnD))])
cyl_lnD_stds = np.array([unc.std_dev(cyl_lnD[i]) for i in range(0, len(cyl_lnD))])

plane_lnD_vals = np.array([unc.nominal_value(plane_lnD[i]) for i in range(0, len(plane_lnD))])
plane_lnD_stds = np.array([unc.std_dev(plane_lnD[i]) for i in range(0, len(plane_lnD))])

sphere_lnD_vals = np.array([unc.nominal_value(sphere_lnD[i]) for i in range(0, len(sphere_lnD))])
sphere_lnD_stds = np.array([unc.std_dev(sphere_lnD[i]) for i in range(0, len(sphere_lnD))])

### Run regression, generate plots, and print diffusion parameters to console (currently not set up to calculate plane)
sphere_lnDo_unc, sphere_Ea_unc, cyl_lnDo_unc, cyl_Ea_unc = Arrplot(T_1000_vals, T_1000_stds, sphere_lnD_vals, sphere_lnD_stds, T_1000_vals, T_1000_stds, cyl_lnD_vals, cyl_lnD_stds)
print('Sphere Geometry: ln(Do) = ' + str(sphere_lnDo_unc) + ', Ea = ' + str(sphere_Ea_unc))
sphere_Do = umath.exp(sphere_lnDo_unc)
print('Sphere Do = ' + str(sphere_Do))
print('Cylinder Geometry: ln(Do) = ' + str(cyl_lnDo_unc) + ', Ea = ' + str(cyl_Ea_unc))
cyl_Do = umath.exp(cyl_lnDo_unc)
print('Cylinder Do = ' + str(cyl_Do))

### End Code