import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import math
import pdb
import scipy.interpolate as interpolate
import os.path
from scipy.misc import derivative
from astropy.stats import sigma_clip
from numpy.polynomial import Polynomial
import sympy
from sympy import oo

"""
for every visit containing a flare, 
see if there's an fuv file for it
if so calculate the color at each point
evaluate the polynomial convenience functions at those points
to get the temperature of the flare 
"""

# plot parameters
mpl.rcParams['axes.linewidth'] = 1.75
plt.rc('font',family='serif')
mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rcParams['agg.path.chunksize'] = 2000
plt.figure(figsize=(12, 8))
plt.style.use(['science', 'no-latex'])
plt.minorticks_on()
plt.tick_params(direction='in', length=8, width=1, which='major', labelsize=21, top=True, right=True)
plt.tick_params(direction='in', length=4, width=1, which='minor', labelsize=21, top=True, right=True)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)


lum_file = open('luminosities.txt', 'w')
energy_file = open('energies.txt', 'w')
ratio_file = open('ratio_fuv_nuv.txt', 'w')

# open fuv source name file
# fuv_names = open('fuv_source_names.txt', 'r')
# fuv_names = fuv_names.readlines() # of form id.csv
# fuv_names = [f.strip()[:-4] for f in fuv_names] #  gaia id of fuv sources

# open list of filenames for nuv visits with detected flares
detected_fnames = open('detected_possibly_final.txt', 'r')
detected_fnames = detected_fnames.readlines()
# gaiaID-NUV#
detected_fnames = [f.strip()[:-4] for f in detected_fnames]
# Gaia ID
detected_roots = [f.split('-')[0] for f in detected_fnames]

# ids = set(detected_roots).intersection(fuv_names)

# open csv containing GCNS data for the sources in our sample
data = pd.read_csv('source_info.csv', skiprows=0, header=0, sep=' ', index_col=False)
data['source'] = pd.to_numeric(data.source, errors='coerce')
data['dist'] = pd.to_numeric(data.dist_kpc, errors='coerce')
data['source'] = data['source'].astype(str)

# ********** bb and temp func ********** 
# global variables
h = 6.626e-34
c = 2.998e+8
k = 1.38e-23
# from filter profile service 
lambda_ref_f = 1535.08 * 1e-10
lambda_ref_n = 2300.78 * 1e-10

# load filter response functions
fuv_ftc = np.genfromtxt('/Users/veraberger/reu/GALEX_GALEX.FUV.dat') 
nuv_ftc = np.genfromtxt('/Users/veraberger/reu/GALEX_GALEX.NUV.dat')

# planck function. takes wavelength & temperature, returns bb intensity
def planck(wav, T):
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity

# generate x-axis for bb curve in increments from 1nm to 3 micrometer in 1 nm increments
# starting at 1 nm to avoid wav = 0, which would result in division by zero.
wavelengths = np.arange(1e-9, 3e-6, 1e-9) # meters

# initialize color and temperature arrays
colorArr = []
tempArr = []
ratioArr = []
# generate colors spanning a range of temperatures
for t in range(1000, 300000, 20):
    bb_intensity = planck(wavelengths, t)
    
    # get wavelengths and transmissions from filter response functions
    nuv_fc_lams = nuv_ftc*1e-10
    nuv_fc_trans = nuv_ftc[:,1]
    nuv_intensity = np.interp(nuv_fc_lams, wavelengths, bb_intensity) # returns bb sed at wavelengths given by filter response func
    # synthetic photometry - integrate over range of wavelengths 
    # https://mfouesneau.github.io/pyphot/photometry.html#ab-magnitude-system
    nuv_top = np.trapz(nuv_intensity * nuv_fc_trans / nuv_fc_lams, nuv_fc_lams) 
    nuv_bottom = np.trapz(nuv_fc_trans / nuv_fc_lams, nuv_fc_lams)
    flux_nuv = nuv_top / nuv_bottom
    fuv_fc_lams = fuv_ftc*1e-10
    fuv_fc_trans = fuv_ftc[:,1]
    fuv_intensity = np.interp(fuv_fc_lams, wavelengths, bb_intensity) 
    fuv_top = np.trapz(fuv_intensity * fuv_fc_trans /fuv_fc_lams, fuv_fc_lams)
    fuv_bottom = np.trapz(fuv_fc_trans / fuv_fc_lams, fuv_fc_lams)
    flux_fuv = fuv_top / fuv_bottom
    ratioArr.append(flux_fuv / flux_nuv)
    
    # convert fluxes to magnitudes
    mag_f = -2.5 * np.log10(flux_fuv) - 2.5*np.log10(lambda_ref_f**2 / c) - 48.60
    mag_n = -2.5 * np.log10(flux_nuv) - 2.5*np.log10(lambda_ref_n**2 / c) - 48.60

    # compute FUV - NUV color
    color = mag_f - mag_n
    colorArr.append(color)
    tempArr.append(t)

# turn lists into arrays
tempArr = np.array(tempArr)
colorArr = np.array(colorArr)
ratioArr = np.array(ratioArr)

# create interpolating function, returns temperature given an array of FUV-NUV colors
temp_func = interpolate.interp1d(colorArr, tempArr, bounds_error=False, fill_value='extrapolate')

# plot ratios
plt.plot(temp_func(colorArr), ratioArr, color='#550527', alpha=0.7, linewidth=3)
plt.yscale('log')
plt.xlim(1000,60000)
plt.ylim(1e-2, 12)
# plt.axhline(3)
# plt.axhline(4.32001503)
n = 10**(.712/2.5)
print(n)
print(sympy.limit(ratioArr, temp_func(colorArr), 10000, '-'))
# plt.axhline(n)
plt.ylabel('FUV/NUV flux ratio', size=24)

plt.axhline(4.328, color='red', label = 'asymptote 4.328')

plt.axhline(3.53, color='orange', label = 'visible flatline 3.53')
plt.legend()
# plt.savefig('figures/october_color_temp_ratio.png', dpi=200)
# plt.show()

# plot fuv-nuv color
# plt.plot(temp_func(colorArr), colorArr, color='#550527', alpha=0.7, linewidth=2)
# plt.xticks([10000, 20000, 30000, 40000, 50000], ['10,000', '20,000', '30,000', '40,000', '50,000'])
# plt.ylabel('FUV - NUV [AB mag]', size=24)
# plt.ylim(0, 4.4)
# plt.xlim(0, 300000)
# plt.xlabel('Temperature [K]', size=24)
# plt.scatter(tempArr, colorArr, s=1, c='#FAA613')
# plt.savefig('figures/color_temp_ratio.png', dpi=200)
# plt.show()



# initialize counts 
count = 0 # no fuv data
i = 0 # files
# detected_fnames = ['4933592687688141056-NUV-9']

for file in detected_fnames:
    i+=1
    nuv_file = file+'.csv'
    fuv_file = file.split('-')[0]+'-FUV'+'.csv'
    dataname = file.split('-')[0]
    # dataname = '1461125613285603840'
    dist = data['dist_kpc']
    dist_kpc = dist[data['source'] == dataname] 
    dist_cm = dist_kpc * 3.085677581e21
    
    if True == True:
    # if the source has FUV data, load NUV visit containing a flare and corresponding FUV data
    # if os.path.exists('LCs/'+fuv_file) & os.path.exists('1sigma_flare_visits/'+nuv_file):
        
        # read in files
        df_nuv = pd.read_csv('flare_visits_nuv2/'+nuv_file, skiprows=0, header=0, sep=',', index_col=False)
        df_fuv = pd.read_csv('LCs/'+fuv_file, skiprows=0, header=0, sep=',', index_col=False)


        # read in columns
        df_nuv['flux'] = pd.to_numeric(df_nuv.flux_bgsub, errors='coerce')
        df_nuv['ferr'] = pd.to_numeric(df_nuv.flux_bgsub_err, errors='coerce')
        df_nuv['t0'] = pd.to_numeric(df_nuv.t0, errors='coerce')
        df_nuv['t1'] = pd.to_numeric(df_nuv.t1, errors='coerce')
        df_nuv['flare'] = pd.to_numeric(df_nuv.flare, errors='coerce')

        # make new t0 column and turn the original into the index
        df_nuv['nt0'] = df_nuv['t0']
        df_nuv = df_nuv.set_index('t0')

        # first and last times of exposure in NUV visit to constrict FUV data
        t_init = df_nuv['nt0'].iloc[0] 
        t_final = df_nuv['t1'].iloc[-1]
        df_fuv['ft0'] = df_fuv['t0']
        df_fuv['t1'] = pd.to_numeric(df_fuv.t1, errors='coerce')
        df_fuv = df_fuv.loc[(df_fuv['t0'] >= t_init) & (df_fuv['t0'] <= t_final)]

        df_fuv['flux'] = pd.to_numeric(df_fuv.flux_bgsub, errors='coerce')
        df_fuv['ferr'] = pd.to_numeric(df_fuv.flux_bgsub_err, errors='coerce')
        df_fuv['t0'] = pd.to_numeric(df_fuv.t0, errors='coerce')
        df_fuv = df_fuv.rename(columns={'flags': 'flgs'})
        df_fuv['flgs'] = pd.to_numeric(df_fuv.flgs, errors='coerce')
        flags = df_fuv['flgs']
        # eliminate outlier points
        df_fuv = df_fuv.loc[(df_fuv['flux'] > -10e-5) &  (df_fuv['flux'] < 10e-5 )]
        # eliminate points with worst flags
        df_fuv = df_fuv.loc[(flags != 1.0) & (flags != 2.0) & (flags != 64.0) &(flags != 128.0) &(flags != 256.0) &(flags != 512.0) ]
        df_fuv = df_fuv.set_index('t0')

        # align NUV and FUV dataframes by timestamp
        df_nuv, df_fuv = df_nuv.align(df_fuv, join="outer", axis=0, fill_value=None)
        # remove rows with nan timestamps (times that don't overlap btw nuv and fuv)
        df_nuv = df_nuv.loc[df_nuv.index.dropna()]
        df_fuv = df_fuv.loc[df_fuv.index.dropna()]

        flux_nuv = df_nuv['flux']
        nuv_ferr = df_nuv['ferr']
        tn_mean = (df_nuv['nt0'] + df_nuv['t1'])/2 # average time in each bin
        tn_mean = ((tn_mean+ 315964800) / 86400) + 40587 # converts to MJD
        flux_fuv = df_fuv['flux']
        fuv_ferr = df_fuv['ferr']
        tf_mean = (df_fuv['ft0'] + df_fuv['t1'])/2
        tf_mean = ((tf_mean+ 315964800) / 86400) + 40587
        init_time = math.floor(tn_mean.iloc[0])
        if flux_fuv.isnull().values.all():
            count+=1
            # df_fuv = pd.read_csv('LCs/'+fuv_file, skiprows=0, header=0, sep=',', index_col=False)
            # print(df_fuv)
            # df_fuv['t1'] = pd.to_numeric(df_fuv.t1, errors='coerce')
            # df_fuv['flux'] = pd.to_numeric(df_fuv.flux_bgsub, errors='coerce')
            # df_nuv = pd.read_csv('1sigma_flare_visits/'+nuv_file, skiprows=0, header=0, sep=',', index_col=False)
            # df_nuv['t1'] = pd.to_numeric(df_nuv.t1, errors='coerce')
            # df_nuv['flux'] = pd.to_numeric(df_nuv.flux_bgsub, errors='coerce')
            # df2 = pd.read_csv('LCs/'+file.split('-')[0]+'-NUV'+'.csv', skiprows=0, header=0, sep=',', index_col=False)
            # df2['t1'] = pd.to_numeric(df_nuv.t1, errors='coerce')
            # df2['flux'] = pd.to_numeric(df_nuv.flux_bgsub, errors='coerce')
            # plt.errorbar(df2['t1'], df2['flux'], ls='none', marker='s', color='lightpink', markersize=4, alpha=0.7)
            # plt.errorbar(df_fuv['t1'], df_fuv['flux'], ls='none', marker='o', color='blue', markersize=2)
            # plt.errorbar(df_nuv['t1'], df_nuv['flux'], ls='none', marker='o', color='red', markersize=2)
            


        # do sigma clipping and subtract baseline flux before converting to mag

        with plt.rc_context({'font.family':'serif', 'axes.labelsize':19}):
            fig, ax = plt.subplots(2,1, figsize=(10,15), sharex=True)
            plt.style.use(['science', 'no-latex'])
            mpl.rcParams['axes.linewidth'] = 1.75
            plt.rc('font',family='serif')
            mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
            plt.rcParams['agg.path.chunksize'] = 2000
            ax[0].tick_params(direction='out', length=5, width=1, which='both', labelsize=18, top=True, right=True)
            ax[1].tick_params(direction='out', length=5, width=1, which='both', labelsize=18, top=True, right=True)

        #ax[0].errorbar(tf_mean, flux_fuv, yerr=fuv_ferr, color='orange', ls='none', markersize=6, marker='o', label='FUV pre nonlinearity corr')
        #ax[0].errorbar(tn_mean, flux_nuv, yerr=fuv_ferr, color='yellow', ls='none', markersize=6, marker='o', label='NUV pre nonlinearity corr')
        
        # nonlinearity -- when it's too bright, you have fewer counts than you'd expect and need to correct for that
        # upper limits - if flux is less than 3x its error, plot it as an upper limit
        for m in df_fuv.index:
            if flux_fuv[m] > 1.97e-14:
                mag_m= -2.5 * np.log10(flux_fuv[m]/(1.40e-15 )) + 18.81
                mag_corr = 6.412 + ((17.63 * mag_m- 192.135) ** 0.5)
                flux_fuv[m] = 1.4e-15 * 10 ** ((-mag_corr + 18.82) / 2.5)
            # if flux_fuv[m] < 3 * fuv_ferr[m]: 
            #     flux_fuv[m] = 3 * fuv_ferr[m]
        for m in df_nuv.index:
            if flux_nuv[m] > 3.68e-15:
                mag_m = -2.5 * np.log10(flux_nuv[m]/(2.06e-16 )) + 20.08
                mag_corr = 3.778 + ((24.337 * mag_m - 241.018) ** 0.5)
                flux_nuv[m] = 2.06e-16 * 10 ** ((-mag_corr + 20.08) / 2.5)
            # if flux_nuv[m] < 3 * nuv_ferr[m]: 
                # flux_nuv[m] = 3 * nuv_ferr[m]
       
        # compute subtract baseline flux from each lightcurve
        # sigma clip to remove flare from median flux calculation
        clipped_flux_nuv = sigma_clip(flux_nuv, sigma=2, cenfunc='median', stdfunc='std', masked=False)
        clipped_flux_fuv = sigma_clip(flux_fuv, sigma=2, cenfunc='median', stdfunc='std', masked=False) 
        std_nuv = np.std(clipped_flux_nuv)
        std_fuv = np.std(clipped_flux_fuv)
        baseline_nuv = np.median(clipped_flux_nuv)
        baseline_fuv = np.median(clipped_flux_fuv)
        flux_nuv -= baseline_nuv
        flux_fuv -= baseline_fuv

        # convert fluxes to luminosities
        lum_params_nuv = 4 * np.pi * (dist_cm ** 2) * 2267
        lum_params_fuv = 4 * np.pi * (dist_cm ** 2) * 1516
        lum_nuv = flux_nuv.apply(lambda x: x* lum_params_nuv) 
        lum_fuv = flux_fuv.apply(lambda x: x* lum_params_fuv) 
        lum_nuv = lum_nuv.squeeze()
        lum_fuv = lum_fuv.squeeze()

        lum_n_err = nuv_ferr.apply(lambda x: x * lum_params_nuv) 
        lum_f_err = fuv_ferr.apply(lambda x: x * lum_params_fuv) 
        lum_n_err = lum_n_err.squeeze()
        lum_f_err = lum_f_err.squeeze()

        lum_max = max(np.nanmax(lum_fuv), np.nanmax(lum_nuv))
        log_max_lum = np.log10(lum_max)
        try:  
            log_max_lum = math.floor(log_max_lum)
        except:
            try:
                log_max_lum = math.floor(lum_fuv.max())
            except:
                try:
                    log_max_lum =  math.floor(lum_nuv.max())
                except: log_max_lum = 29
        if (log_max_lum > 38) or (log_max_lum is None):
            log_max_lum = 38
            print(df_nuv.head())

        
        print(log_max_lum)

        # compute flare UV energies
        bin_length = df_nuv['t1'].iloc[0] - df_nuv['nt0'].iloc[0]
        energy_n = (lum_nuv * bin_length).sum()
        # energy_f = (lum_fuv * bin_length).sum()
        # energy_file.write(file+' '+str(energy_n)+' '+str(energy_f)+'\n')

        df_nuv['mag_nuv_app'] = -2.5 * np.log10(flux_nuv/(2.06e-16 )) + 20.08
        df_nuv['mag_fuv_app']= -2.5 * np.log10(flux_fuv/(1.40e-15 )) + 18.81
        mag_nuv = df_nuv['mag_nuv_app']
        mag_fuv = df_nuv['mag_fuv_app']

        #df_nuv['color'] = mag_fuv - mag_nuv
        mag_fuv = mag_fuv.interpolate()
        mag_err_nuv = nuv_ferr / flux_nuv / 0.92
        mag_err_fuv = fuv_ferr / flux_fuv / 0.92

        flare_nuv_fluxes = flux_nuv.loc[df_nuv['flare'] == 1]
        flare_fuv_fluxes = flux_fuv.loc[df_nuv['flare'] == 1]
        flare_mag_err_nuv =  mag_err_nuv.loc[df_nuv['flare'] == 1]
        flare_mag_err_fuv =  mag_err_fuv.loc[df_nuv['flare'] == 1]

        flare_lum_nuv = lum_nuv.loc[df_nuv['flare'] == 1]
        flare_lum_fuv = lum_fuv.loc[df_nuv['flare'] == 1]
        # max_flux = max(np.nanmax(flux_fuv), np.nanmax(flux_nuv))
        # log_max_flux = np.log10(max_flux)
        # try:
        #     log_max_flux = math.floor(log_max_flux)
        # except: 
        #     print(file)

        max_flux = max(np.nanmax(flux_fuv), np.nanmax(flux_nuv))
        log_max_flux = np.log10(max_flux)
        try:  
            log_max_flux = math.floor(log_max_flux)
        except:
            try:
                log_max_flux = math.floor(flux_fuv.max())
            except:
                try:
                    log_max_flux =  math.floor(flux_nuv.max())
                except: log_max_flux = 29
        if (log_max_flux > 38) or (log_max_flux is None):
            log_max_flux= -16


        # max_ratio = np.nanmax(flare_fuv_fluxes / flare_nuv_fluxes)
        # ratio_err = np.sqrt((nuv_ferr / flux_nuv) ** 2 + (fuv_ferr / flux_fuv) ** 2)
        # #max_rerr = np.nanmax(ratio_err) redo this s.t. the error you compute is on the single max ratio - find 
        # # find the fluxes associated w the max ratio
        # ratio_file.write(file+' '+str(max_ratio)+' '+str(ratio_err)+'\n')
        # print('got ratio')

  
        flare_pts = tn_mean.loc[df_nuv['flare'] == 1]
        start = flare_pts.iloc[0]
        end = flare_pts.iloc[-1]
        # shaded region around flare
        ax[0].axvspan(0.68916, 0.6913, color='#D2F4B8', alpha=0.5, lw=0)
        ax[1].axvspan(0.68916, 0.6913, color='#D2F4B8', alpha=0.5, lw=0)
        # ax[0].axvspan(start-init_time, end-init_time, color='#D2F4B8', alpha=0.5, lw=0)
        # ax[1].axvspan(start-init_time, end-init_time, color='#D2F4B8', alpha=0.5, lw=0)
        df_nuv['color'] = -2.5 * np.log10(flux_fuv / flux_nuv)
        color = -2.5 * np.log10(flare_fuv_fluxes / flare_nuv_fluxes)
        #color = mag_fuv - mag_nuv

        y = temp_func(color)
        df_nuv['all_temps'] = temp_func(df_nuv['color'])

        # check by looking at temps corresponding to lower and upper limits for color
        x_err = np.sqrt(flare_mag_err_nuv**2 + flare_mag_err_fuv**2)
        y_err = x_err * derivative(temp_func, color)
        #Polynomial.fit(flare_pts, y, 2)


        # plot temperature
        ax[1].errorbar(flare_pts-init_time,y, yerr=y_err, ls='none', color='#550527', markeredgecolor='chocolate', marker='o', markersize=5, markeredgewidth=0.6, capsize=3)
        

        # # plot in luminosity
        # # ax[0].errorbar(tf_mean-init_time, lum_fuv/10**log_max_lum, yerr=lum_f_err/10**log_max_lum, ls='none', color='#3891A6', marker='o', markersize=5, markeredgewidth=0.4, capsize=3, label='FUV' )
        # # ax[0].errorbar(tn_mean-init_time, lum_nuv/10**log_max_lum, yerr=lum_n_err/10**log_max_lum, ls='none', color='#BF3100', marker='s', markersize=5, markeredgewidth=0.4, capsize=3, label='NUV'  )
        # # ax[0].set_ylabel('Luminosity [1e+'+str(log_max_lum)+' erg $s^{-1} $]', size = 18)


        # # plot in flux
        ax[0].errorbar(tf_mean-init_time, flux_fuv/10**log_max_flux, yerr=fuv_ferr/10**log_max_flux, ls='none', color='#3891A6', marker='o', markersize=3, markeredgewidth=0.4, capsize=3, label='FUV' )
        ax[0].errorbar(tn_mean-init_time, flux_nuv/10**log_max_flux, yerr=nuv_ferr/10**log_max_flux, ls='none', color='#BF3100', marker='s', markersize=3, markeredgewidth=0.4, capsize=3, label='NUV'  )
        ax[0].set_ylabel('Flux [1e'+str(log_max_flux)+' erg $\; s^{-1} cm^{-2} \\AA ^{-1}$]', size = 18)


        """
        # plot in mag
        ax[0].errorbar(tf_mean.loc[flux_fuv >= lim_fuv]-init_time, mag_fuv.loc[flux_fuv >= lim_fuv],  ls='none', color='#3891A6', marker='o', markersize=4, markeredgewidth=0.6, capsize=3, label='FUV' )
        ax[0].errorbar(tn_mean.loc[flux_nuv >= lim_nuv]-init_time, mag_nuv.loc[flux_nuv >= lim_nuv],  ls='none', color='#BF3100', marker='o', markersize=4, markeredgewidth=0.6, capsize=3, label='NUV'  )
        ax[0].errorbar(tf_mean.loc[flux_fuv < lim_fuv]-init_time, upperlim_fuv,  ls='none', color='#3891A6', marker='v', markersize=4, markeredgewidth=0.6, capsize=3)
        ax[0].errorbar(tn_mean.loc[flux_nuv < lim_nuv]-init_time, upperlim_nuv, ls='none', color='#BF3100', marker='v', markersize=4, markeredgewidth=0.6, capsize=3)
        ax[0].invert_yaxis()
        ax[0].set_ylabel('AB Magnitude [mag]', size = 18)
        """

        # ax[1].set_ylim(0, 50000)
        # # ax[1].set_yticks([0, 20000, 40000, 60000, 80000, 100000], ['0', '20,000', '40,000', '60,000', '80,000', '100,000'])
        # ax[1].set_xlabel('MJD -'+str(init_time), size=18)
        # ax[1].set_ylabel('Temperature [K]', size=18)
        # # ax[0].set_title(file.split('-')[0]+'-'+file.split('-')[2], size=20)
        # ax[0].legend(fontsize=20)
        # ax[0].set_xlim(0.688, 0.693)
        # plt.subplots_adjust(hspace=.0) # make the plots touch
        # plt.savefig('figures/transparent/'+file+'_1s.png', dpi=200, bbox_inches='tight', transparent=True)
        # # plt.savefig('figures/temps_6/'+file+'.png', dpi=200, bbox_inches='tight')

        plt.show()
        plt.clf() 


        lum_file.write(file+' '+str(lum_nuv.max())+'\n')

        newdf = pd.DataFrame()
        flare_data = df_nuv.loc[df_nuv['flare'] == 1]
        newdf['t0'] = flare_data['nt0']
        newdf['t1'] =  flare_data['t1']
        newdf['t_mean_mjd'] = tn_mean
        newdf['t_mean_s'] = (flare_data['nt0'] +flare_data['t1']) / 2
        newdf['flux_n'] =  flare_nuv_fluxes
        newdf['ferr_n'] = nuv_ferr.loc[df_nuv['flare'] == 1]
        newdf['flux_f'] =  flare_fuv_fluxes
        newdf['ferr_f'] = fuv_ferr.loc[df_nuv['flare'] == 1]
        newdf['mag_n'] = mag_nuv.loc[df_nuv['flare'] == 1]
        newdf['mag_f'] = mag_fuv.loc[df_nuv['flare'] == 1]
        newdf['mag_err_n'] = flare_mag_err_nuv
        newdf['mag_err_f'] = flare_mag_err_fuv
        newdf['color'] = color
        newdf['color_err'] = x_err
        newdf['temp'] = y
        newdf['temp_err'] = y_err
        newdf['lum_n'] = lum_nuv.loc[df_nuv['flare'] == 1]
        newdf['lum_n_err'] = lum_n_err.loc[df_nuv['flare'] == 1]
        newdf['lum_f'] = lum_fuv.loc[df_nuv['flare'] == 1]
        newdf['lum_f_err'] = lum_f_err.loc[df_nuv['flare'] == 1]
        newdf.to_csv('flare_luminosities_oct/'+file+'.csv', sep=',')

    else:
        count += 1   
        # read in files
        # df_nuv = pd.read_csv('1sigma_flare_visits/'+nuv_file, skiprows=0, header=0, sep=',', index_col=False)

        # # read in columns
        # df_nuv['flux'] = pd.to_numeric(df_nuv.flux_bgsub, errors='coerce')
        # df_nuv['ferr'] = pd.to_numeric(df_nuv.flux_bgsub_err, errors='coerce')
        # df_nuv['t0'] = pd.to_numeric(df_nuv.t0, errors='coerce')
        # df_nuv['t1'] = pd.to_numeric(df_nuv.t1, errors='coerce')
        # df_nuv['flare'] = pd.to_numeric(df_nuv.flare, errors='coerce')

        # # make new t0 column and turn the original into the index
        # df_nuv['nt0'] = df_nuv['t0']
        # df_nuv = df_nuv.set_index('t0')

        # # first and last times of exposure in NUV visit to constrict FUV data
        # t_init = df_nuv['nt0'].iloc[0] 
        # t_final = df_nuv['t1'].iloc[-1]

        # flux_nuv = df_nuv['flux']
        # nuv_ferr = df_nuv['ferr']
        # tn_mean = (df_nuv['nt0'] + df_nuv['t1'])/2 # average time per bin
        # tn_mean = ((tn_mean+ 315964800) / 86400) + 40587 # converts to MJD
        # init_time = math.floor(tn_mean.iloc[0])

        # for i in df_nuv.index:
        #     if flux_nuv[i] > 3.68e-15:
        #         mag_i = -2.5 * np.log10(flux_nuv[i]/(2.06e-16 )) + 20.08
        #         mag_corr = 3.778 + ((24.337 * mag_i - 241.018) ** 0.5)
        #         flux_nuv[i] = 2.06e-16 * 10 ** ((-mag_corr + 20.08) / 2.5)
            
        # # compute subtract baseline flux from each lightcurve
        # # sigma clip to remove flare from median flux calculation
        # clipped_flux_nuv = sigma_clip(flux_nuv, sigma=2, cenfunc='median', stdfunc='std', masked=False)
        # std_nuv = np.std(clipped_flux_nuv)
        # baseline_nuv = np.median(clipped_flux_nuv)
        # flux_nuv -= baseline_nuv
        # lum_params_nuv = 4 * np.pi * (dist_cm ** 2) * 2267
        # lum_nuv = flux_nuv.apply(lambda x: x* lum_params_nuv) 
        # lum_nuv = lum_nuv.squeeze()
        # lum_n_err = nuv_ferr.apply(lambda x: x * lum_params_nuv) 
        # lum_n_err = lum_n_err.squeeze()

        # df_nuv['mag_nuv_app'] = -2.5 * np.log10(flux_nuv/(2.06e-16 )) + 20.08
        # mag_nuv = df_nuv['mag_nuv_app']
        # mag_err_nuv = nuv_ferr / flux_nuv / 0.92
        # flare_nuv_fluxes = flux_nuv.loc[df_nuv['flare'] == 1]
        # flare_mag_err_nuv =  mag_err_nuv.loc[df_nuv['flare'] == 1]
        # flare_lum_nuv = lum_nuv.loc[df_nuv['flare'] == 1]

        # flare_pts = tn_mean.loc[df_nuv['flare'] == 1]
        # start = flare_pts.iloc[0]
        # end = flare_pts.iloc[-1]

        # max_flux = np.nanmax(flux_nuv)
        # log_max_flux = np.log10(max_flux)
        # try:
        #     log_max_flux = math.floor(log_max_flux)
        # except: 
        #     print(file)
        # plt.figure(figsize=(12,8))
        # plt.style.use(['science', 'no-latex'])
        # mpl.rcParams['axes.linewidth'] = 1.75
        # plt.rc('font',family='serif')
        # mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
        # plt.rcParams['agg.path.chunksize'] = 2000
        # plt.tick_params(direction='in', length=5, width=1, which='both', labelsize=18, top=True, right=True)

        # plt.errorbar(tn_mean-init_time, flux_nuv*10**log_max_flux, yerr=nuv_ferr*10**log_max_flux, ls='none', color='#BF3100', marker='s', markersize=5, markeredgewidth=0.4, capsize=3, label='NUV'  )
        # plt.axvspan(start-init_time, end-init_time, color='#D2F4B8', alpha=0.5, lw=0)
        # plt.ylabel('Flux [1e+'+str(log_max_flux)+' erg $\; s^{-1} cm^{-2} \\AA ^{-1}$]', size = 18)
        # plt.xlabel('MJD -'+str(init_time), size=18)
        # plt.legend( fontsize=20)
        # # plt.show()
        # plt.savefig('figures/temps_6/'+file+'.png', dpi=200)

         

        # newdf = pd.DataFrame()
        # flare_data = df_nuv.loc[df_nuv['flare'] == 1]
        # newdf['t0'] = flare_data['nt0']
        # newdf['t1'] =  flare_data['t1']
        # newdf['t_mean_mjd'] = tn_mean
        # newdf['t_mean_s'] = (flare_data['nt0'] +flare_data['t1']) / 2
        # newdf['flux_n'] =  flare_nuv_fluxes
        # newdf['ferr_n'] = nuv_ferr.loc[df_nuv['flare'] == 1]
        # newdf['flux_f'] =  np.nan
        # newdf['ferr_f'] = np.nan
        # newdf['mag_n'] = mag_nuv.loc[df_nuv['flare'] == 1]
        # newdf['mag_f'] = np.nan
        # newdf['mag_err_n'] = flare_mag_err_nuv
        # newdf['mag_err_f'] = np.nan
        # newdf['color'] = np.nan
        # newdf['color_err'] = np.nan
        # newdf['temp'] = np.nan
        # newdf['temp_err'] = np.nan
        # newdf['lum_n'] = lum_nuv.loc[df_nuv['flare'] == 1]
        # newdf['lum_n_err'] = lum_n_err.loc[df_nuv['flare'] == 1]
        # newdf['lum_f'] = np.nan
        # newdf['lum_f_err'] = np.nan
        # newdf.to_csv('flare_data/'+file+'.csv', sep=',')


print(count)



lum_file.close()
ratio_file.close()








