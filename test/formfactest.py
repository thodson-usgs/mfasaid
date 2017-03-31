import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

import dualfrequency

# acoustic frequency
FREQUENCY = 1e6  # Hz

# acoustic wavelength in meters
WAVELENGTH = (dualfrequency.SPEED_OF_SOUND_IN_WATER / FREQUENCY)

# acoustic wave number, in 1/meters
WAVE_NUMBER = 2 * np.pi / WAVELENGTH

KAPPA = 0.4


def formfun_frac_rayleigh(kappa):

    dividend = 1 + 15*kappa**2 + 45*kappa**4 + 15*kappa**6
    divisor = 1 + 3*kappa**2

    mean_fe = np.sqrt( dividend / divisor )

    return mean_fe


def formfun_frac_scattering(kappa):

    dividend = 1 + kappa**2
    divisor = 1 + 3*kappa**2

    mean_fe = np.sqrt( dividend / divisor )

    return mean_fe


def get_lognormal_dist(mean, sd):

    sdlog = np.sqrt(np.log((sd/mean)**2 + 1))

    meanlog = np.log(mean) - 0.5*sdlog**2

    dist = lognorm(s=sdlog, loc=0, scale=np.exp(meanlog))

    return dist


def get_vol_cdf_from_num_cdf(d_cdf, num_cdf):

    d_diff = np.diff(d_cdf)
    d_pdf = d_cdf[:-1] + d_diff/2

    vol = 4/3*np.pi*(d_pdf/2)**3
    num_frac = np.diff(num_cdf)

    vol_frac = num_frac*vol/np.sum(num_frac*vol)

    vol_cdf = np.repeat(np.nan, num_cdf.shape)

    vol_cdf[0] = 1 - np.sum(vol_frac)
    vol_cdf[1:] = np.cumsum(vol_frac) + vol_cdf[0]

    return vol_cdf


def norm_analy_form_function(x, kappa):

    f_e = dualfrequency.SedimentSizeDistribution.form_function(x)

    mean_f_e = np.repeat(np.nan, x.shape)

    rayleigh_index = x < 5.5e-1
    mean_f_e[rayleigh_index] = formfun_frac_rayleigh(kappa) * f_e[rayleigh_index]

    scattering_index = x > 2.5
    mean_f_e[scattering_index] = formfun_frac_scattering(kappa) * f_e[scattering_index]

    return mean_f_e


def normal_dist(a, mu_a, sigma_a):

    P_N = 1/(sigma_a*np.sqrt(2*np.pi))*np.exp(-(a-mu_a)**2/(2*sigma_a**2))

    return P_N


def create_thorne_plot():

    # dimensionless wave number
    x = np.logspace(-2, 2, num=1000)

    # particle radius, in meters
    a = x / WAVE_NUMBER

    f_e = dualfrequency.SedimentSizeDistribution.form_function(x)
    plt.loglog(x, f_e, 'k-', label='Uniform grain size')

    mean_f_e_N = np.repeat(np.nan, x.shape)
    mean_f_e_lnN = np.repeat(np.nan, x.shape)

    for mu_a in a:

        index = mu_a == a
        sigma_a = KAPPA * mu_a

        prob_dist = normal_dist(a, mu_a, sigma_a)
        mean_f_e_N[index] = dualfrequency.SedimentSizeDistribution.mean_form_function(a, prob_dist, f_e)

        lnN_dist = get_lognormal_dist(mu_a, sigma_a)
        prob_dist = lnN_dist.pdf(a)
        mean_f_e_lnN[index] = dualfrequency.SedimentSizeDistribution.mean_form_function(a, prob_dist, f_e)

    plt.loglog(x, mean_f_e_N, 'k--', label='Normal distribution')
    plt.loglog(x, mean_f_e_lnN, 'k:', label='Lognormal distribution')

    x = np.logspace(-2, 2, num=50)
    norm_f_e = norm_analy_form_function(x, KAPPA)
    plt.loglog(x[x < 5.5e-1], norm_f_e[x < 5.5e-1], 'k+', label='Rayleigh regime')
    plt.loglog(x[x > 2.5], norm_f_e[x > 2.5], 'ko', markerfacecolor='none', label='Scattering regime')

    plt.gca().tick_params(direction='in', which='both', top=True, bottom=True, left=True, right=True)
    plt.legend(loc='best')
    plt.xlim(1e-1, 2e1)
    plt.ylim(1e-2, 2)
    plt.xlabel('<x>')
    plt.ylabel('f')

    return plt.gca()

# dimensionless wave number
dimensionless_wave_numbers = np.logspace(-2, 2, num=25)
mean_form_function = np.zeros(dimensionless_wave_numbers.shape)

for x in dimensionless_wave_numbers:

    x_index = x == dimensionless_wave_numbers

    # particle radius, in meters
    a = x / WAVE_NUMBER

    # mean diameter
    mean_diameter = 2*a*1000

    # standard deviation
    std = KAPPA*mean_diameter

    ln_N_dist = get_lognormal_dist(mean_diameter, std)

    alpha = 0.0001

    d_low_quantile = ln_N_dist.ppf(alpha)
    d_high_quantile = ln_N_dist.ppf(1-alpha)

    d_cdf = np.linspace(d_low_quantile, d_high_quantile, num=int(1e6))

    num_cdf = ln_N_dist.cdf(d_cdf)
    vol_cdf = get_vol_cdf_from_num_cdf(d_cdf, num_cdf)

    sed_distribution = dualfrequency.SedimentSizeDistribution(d_cdf, vol_cdf)

    mean_form_function[x_index] = sed_distribution.calc_form_function(FREQUENCY/1000)

ax = create_thorne_plot()

ax.plot(dimensionless_wave_numbers, mean_form_function, 'x')

plt.show()
