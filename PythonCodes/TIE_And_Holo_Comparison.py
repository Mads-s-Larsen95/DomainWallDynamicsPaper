# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 09:30:26 2024

@author: maslar
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.optimize as opt
from scipy.signal import find_peaks
from matplotlib.pyplot import cm
import natsort
#%%
#compare to holo
plt.close("all")
e_const = 1.602e-19
hbar = 1.05e-34
# u0 = 1.256*1e-6
M0 = 1.35 #T
B0 = M0
t0 = 140e-9
path_holo = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\Off_Axis_EH\LineProfiles_Phase\Location_1'
files_holo = glob.glob(path_holo+"\\**.npy")
holo_phi = np.load(files_holo[20])
scale_phi = np.load(files_holo[-1])
holo_x = np.linspace(0,len(holo_phi),len(holo_phi))*scale_phi

path_tie = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\InLine_EH\Linescan\Location_1'
files_tie = glob.glob(path_tie+"\\**.npy")
files_tie = natsort.natsorted(files_tie)
tie_phi = []
defoc_vals = []
for file in files_tie:
    if "Scale" in file:
        scale_tie = np.load(file)
        scale_tie = scale_tie*1e9 #to nm
    else:
        tie_phi.append(np.load(file))
        defoc_vals.append(int(file.split("_um")[0].split("_")[-1]))
        
colors = cm.jet(np.linspace(0,1,len(defoc_vals)))
fig,axs = plt.subplots(tight_layout=True,figsize=(8,6))
for i in range(len(defoc_vals)):
    if i>=0:
        tie_x = np.linspace(0,len(tie_phi[0]),len(tie_phi[0]))*scale_tie
        p = tie_phi[i]
        
        bck = np.polyfit(tie_x,p,2)
        p = p- np.polyval(bck,tie_x)
        axs.plot(tie_x,p,color=colors[i],label="f={:.0f} um".format(defoc_vals[i]),lw=2)
        # axs.plot(tie_x,np.polyval(bck,tie_x)-np.polyval(bck,tie_x)[0],color=colors[i],ls="dashed")
axs.set_xlabel("Distance [nm]",fontsize=20)
axs.set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
axs.tick_params(axis="both",labelsize=14)
axs.legend(loc="upper left",bbox_to_anchor=(1.05,1),fontsize=14)
axs.set_xlim([tie_x[0],tie_x[-1]])
axs.text(-0.05,0.99,"(b)",horizontalalignment="right",verticalalignment="top",transform=axs.transAxes,fontsize=30,fontname="Arial")
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images'

# fig.savefig(save_path+"\\"+"TIE_AllDefocVals.png",bbox_inches="tight")
phi_vec_tie_bcksub = []
defoc_vals_new = []
for i in range(len(defoc_vals)):
    if i > 3:
        tie_x = np.linspace(0,len(tie_phi[0]),len(tie_phi[0]))*scale_tie
        poly = np.polyfit(tie_x,tie_phi[i],2)
        phi_vec_tie = tie_phi[i] - np.polyval(poly,tie_x)
        phi_vec_tie_bcksub.append(phi_vec_tie)
        defoc_vals_new.append(defoc_vals[i])
    # axs.plot(tie_x,tie_phi[i],label="TIE $\Delta \phi (f={:.0f}$um)".format(defoc_vals[i]),lw=2)
fig,axs = plt.subplots(tight_layout=True,figsize=(12,8))
for i in range(len(defoc_vals_new)):
    
    axs.plot(tie_x,phi_vec_tie_bcksub[i],label="TIE $\Delta \phi (f={:.0f}$um)".format(defoc_vals_new[i]),lw=2)
holo_phi = holo_phi-holo_phi[0]
poly_holo = np.polyfit(holo_x,holo_phi,2)
holo_phi = holo_phi - np.polyval(poly_holo,holo_x)
holo_phi = holo_phi
axs.plot(holo_x,holo_phi,color="k",label="Holo $\Delta \phi$",lw=2)
axs.set_xlabel("Distance [nm]",fontsize=20)
axs.set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
axs.tick_params(axis="both",labelsize=14)
axs.legend(loc="upper left",bbox_to_anchor=(1.05,1),fontsize=14)

#%%
def CoshFunc(x,A,x0,sigma,C):
    return A*sigma*np.log( np.cosh( (x-x0) / sigma ) ) + C
scan_dist = 15
lbls = ["(a)","(b)","(c)","(d)","(e)"]
scan_dist_forward = scan_dist -0
scan_dist_backward = scan_dist + 0
plt.close("all")
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images'
fig,axs = plt.subplots(tight_layout=True,figsize=(10,6))
figTIE,axsTIE = plt.subplots(tight_layout=True)
figFFT,axsFFT = plt.subplots(tight_layout=True,figsize=(12,8))
fig2,axs2 = plt.subplots()
colors_tie = cm.jet(np.linspace(0,1,len(defoc_vals)))

sigmas_m = []
sigmas_std = []
amps_m = []
amps_std = []
periodicity = []

LabelNames = []
for i in defoc_vals_new:
    LabelNames.append("Inline EH \n f={:.0f} nm".format(i))
LabelNames.append("Off-Axis EH")

for i in range(len(phi_vec_tie_bcksub)):
    p = phi_vec_tie_bcksub[i]
    p = p-p[0]
    N = 5

    x = tie_x
    axs.plot(x,p+i*np.min(p),color=colors_tie[i],label=LabelNames[i])
    
    axsTIE.plot(x,p,color=colors_tie[i],label=LabelNames[i])
    peaks, peak_properties = find_peaks(p, prominence = 2)#,width=1)
    
    peaks = peaks[peaks>scan_dist]
    
    sigmas = []
    mus = []
    amps = []

    peaks = peaks[peaks>scan_dist]
    for pk in peaks:
        xfit = x[pk-scan_dist_backward:pk+scan_dist_forward]
        N = len(xfit)
        pfit2 = p[pk-scan_dist_backward:pk+scan_dist_forward]

        axs2.plot(xfit,pfit2,color=colors_tie[i])
     
        sigma_fit_param = 50
        amp_fit_param = np.max(pfit2)+np.min(pfit2)
        p0 = [amp_fit_param,xfit[np.argmax(pfit2)],sigma_fit_param,0]

        p1,success = opt.curve_fit(CoshFunc,xfit,pfit2,p0=p0)
        residuals = pfit2- CoshFunc(xfit, *p1)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((pfit2-np.mean(pfit2))**2)
        r_squared = 1 - (ss_res / ss_tot)
        if r_squared > 0.8:
            fit_amp,fit_mu, fit_stdev,C = p1
            fit_amp_2 = fit_amp
        # fit_amp,fit_mu, fit_stdev,C = p1

            sigmas.append(fit_stdev)
            mus.append(fit_mu)
            amps.append(-4*fit_amp)
            axs.plot(xfit,CoshFunc(xfit,*p1)+i*np.min(p),ls="dotted",color=colors_tie[i],alpha=0.6)

    sigmas_m.append(np.mean(sigmas))
    sigmas_std.append(np.std(sigmas))
    amps_m.append(np.mean(amps))
    amps_std.append(np.std(amps))
    nfft = len(p)

    FT = np.fft.fftshift(np.fft.fft(p))#[:nfft//2]
    x_interval = x[1]-x[0]
    
    freqs = np.arange(1/x_interval/-2,1/x_interval/2,1/x_interval/nfft)
    
    FT_norm_abs = abs(FT)/np.max(abs(FT))
    
    FT_Halfed = FT_norm_abs[nfft//2:]
    freqs_Halfed = freqs[nfft//2:]
    idx_max = np.argmax(FT_Halfed[1:])
    idx_max = idx_max+1
    # FT_Max = freqs_Halfed[]
    
    FT_Max = freqs_Halfed[idx_max]

# print(FT_Max)
    axsFFT.plot(freqs_Halfed,FT_Halfed,color=colors[i])
    axsFFT.plot(freqs_Halfed[idx_max],FT_Halfed[idx_max],"kx")
    periodicity.append(1/FT_Max)

holo_phi = holo_phi-holo_phi[0]
axs.plot(holo_x,holo_phi+(i+2)*np.min(p),color="k",label=LabelNames[-1])

scan_dist = 35
scan_dist_forward = scan_dist + 0
scan_dist_backward = scan_dist + 0
peaks, peak_properties = find_peaks(holo_phi, prominence = 2)#,width=1)
    
peaks = peaks[peaks>scan_dist]

sigmas = []
mus = []
amps = []

peaks = peaks[peaks>scan_dist]
for pk in peaks:
    xfit = holo_x[pk-scan_dist_backward:pk+scan_dist_forward]
    N = len(xfit)
    pfit2 = holo_phi[pk-scan_dist_backward:pk+scan_dist_forward]
 
    sigma_fit_param = 50
    amp_fit_param = np.max(pfit2)+np.min(pfit2)
    p0 = [amp_fit_param,xfit[np.argmax(pfit2)],sigma_fit_param,0]

    p1,success = opt.curve_fit(CoshFunc,xfit,pfit2,p0=p0)
    residuals = pfit2- CoshFunc(xfit, *p1)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((pfit2-np.mean(pfit2))**2)
    r_squared = 1 - (ss_res / ss_tot)
    if r_squared > 0.9:
        fit_amp,fit_mu, fit_stdev,C = p1

        sigmas.append(fit_stdev)
        mus.append(fit_mu)
        amps.append(-4*fit_amp)
        
        axs.plot(xfit,CoshFunc(xfit,*p1)+(i+2)*np.min(p),ls="dotted",color="k",alpha=0.6)

sigmas_m.append(np.mean(sigmas))
sigmas_std.append(np.std(sigmas))
amps_m.append(np.mean(amps))
amps_std.append(np.std(amps))

axs.set_xlim([x[0],x[-1]])
axs.set_ylabel("$\Delta \phi $ [rad]",fontsize=30)
axs.set_xlabel("Distance [nm]",fontsize=30)
axs.tick_params(axis="both",labelsize=14)
axs.legend(loc="upper left",bbox_to_anchor=(1.05,1),fontsize=16)

# fig.savefig(save_path+"\\"+"Comparison_IL_OA.png")#,bbox_inches="tight")
FT = np.fft.fftshift(np.fft.fft(holo_phi))#[:nfft//2]
x_interval = holo_x[1]-holo_x[0]
nfft = len(holo_phi)
# FT[0] = 0
freqs = np.arange(1/x_interval/-2,1/x_interval/2,1/x_interval/nfft)

FT_norm_abs = abs(FT)/np.max(abs(FT))

FT_Halfed = FT_norm_abs[nfft//2:]
freqs_Halfed = freqs[nfft//2:]
idx_max = np.argmax(FT_Halfed[1:])
idx_max = idx_max+1
FT_Max = freqs_Halfed[idx_max]

# print(FT_Max)
axsFFT.plot(freqs_Halfed,FT_Halfed,color="k")
axsFFT.plot(freqs_Halfed[idx_max],FT_Halfed[idx_max],"kx")
periodicity.append(1/FT_Max)
# periodicity.append(171.27886885)

x_ticks = np.linspace(1,len(defoc_vals_new),len(defoc_vals_new)+1)
# LabelNames = ["IL-EH \n f={:.0f}" + str(i for i in defoc_vals_new)]
fig,axs = plt.subplots(tight_layout=True,nrows=3,sharex=True,figsize=(10,8))
axs[0].errorbar(x_ticks,sigmas_m,yerr=sigmas_std,capsize=10,color="k",ls="-",marker="o",ms=8)
axs[0].set_ylabel("$\sigma$ [nm]",fontsize=20)
axs[1].errorbar(x_ticks,amps_m,yerr=amps_std,capsize=10,color="k",ls="-",marker="o",ms=8)
axs[1].set_ylabel("A [rad]",fontsize=20)
axs[2].plot(x_ticks,periodicity,ls="-",marker="o",color="k",ms=8)
axs[2].set_ylabel("$\lambda$ [nm]",fontsize=20)
axs[-1].set_xticks(x_ticks)
axs[-1].set_xticklabels(LabelNames)
axs[0].set_ylim([5,20])
axs[1].set_ylim([0,2*5])
axs[-1].set_ylim([150,200])
for n in range(3):
    axs[n].tick_params(axis="both",labelsize=14)
    axs[n].text(-0.05,0.99,lbls[n+1],horizontalalignment="right",verticalalignment="top",transform=axs[n].transAxes,fontsize=30,fontname="Arial")
fig.savefig(save_path+"\\"+"Comparison_IL_OA_Fits.png")
print(hbar*(amps_m[-1])/(e_const*B0*(sigmas_m[-1])*1e-9)*1e9)

#%%
x_ticks = np.linspace(1,len(LabelNames),len(LabelNames))
fig,axs = plt.subplots(tight_layout=True,nrows=4,figsize=(10,8))
for i in range(len(phi_vec_tie_bcksub)):
    p = phi_vec_tie_bcksub[i]
    p = p-p[0]
    N = 5

    x = tie_x
    axs[0].plot(x,p+i*np.min(p),color=colors_tie[i],label=LabelNames[i])
axs[1].errorbar(x_ticks,sigmas_m,yerr=sigmas_std,capsize=10,color="k",ls="-",marker="o")
axs[1].set_ylabel("$\sigma$ [nm]",fontsize=20)
axs[2].errorbar(x_ticks,amps_m,yerr=amps_std,capsize=10,color="k",ls="-",marker="o")
axs[2].set_ylabel("A [rad]",fontsize=20)
axs[3].plot(x_ticks,periodicity,ls="-",marker="o",color="k")
axs[3].set_ylabel("$\lambda$ [nm]",fontsize=20)

axs[3].set_xticklabels(LabelNames)
axs[1].set_ylim([5,20])
axs[2].set_ylim([0,10])
axs[3].set_ylim([150,200])
for n in range(4):
    axs[n].tick_params(axis="both",labelsize=14)
    axs[n].text(-0.05,0.99,lbls[n],horizontalalignment="right",verticalalignment="top",transform=axs[n].transAxes,fontsize=30,fontname="Arial")
    if n>=1:
        axs[n].set_xticks(x_ticks)
fig.savefig(save_path+"\\"+"Comparison_IL_OA_Fits.png")
#%%
plt.close("all")
fig,axs = plt.subplots(tight_layout=True,nrows=2,figsize=(12,10))
for i in range(len(periodicity)):
    axs[0].plot(defoc_vals[i],sigmas_m[i],"o",color=colors_tie[i])
    axs[1].plot(defoc_vals[i],periodicity[i],"o",color=colors_tie[i])
    