# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 11:45:46 2024

@author: maslar
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.optimize as opt
from matplotlib.pyplot import cm
import natsort
#%%
path_holo = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\Off_Axis_EH\LineProfiles_Phase\Location_1'
files_holo = glob.glob(path_holo+"\\**.npy")
files_holo = natsort.natsorted(files_holo)

phi_holo = []
BVal_holo = []
for file in files_holo:
    if "DW" not in file:
        if "Scale" in file:
            scale = np.load(file)
        else:
            phi_holo.append(np.load(file))
            BVal_holo.append(float(file.split("\\")[-1].split("Perc")[0].split("_")[-1])/100*2)
BVal_holo = np.asarray(BVal_holo)
phi_holo = np.asarray(phi_holo)
scale =  np.asarray(scale)   
#%%
def CoshFunc(x,A,x0,sigma,C):
    return A*np.log( np.cosh( (x-x0) / sigma ) )  + C

######################################
save_figs = 0
######################################
e_const = 1.602e-19
hbar = 1.05e-34
u0 = 1.256*1e-6
M0 = 2 #T
B0 = u0*M0
plt.close("all")
lbls = ["(a)","(b)","(c)","(d)","(e)"]
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images'

if "Location_C" in path_holo:
    save_add = "_locC"
if "Location_1" in path_holo:
    save_add = "_loc1"
fig,axs = plt.subplots(tight_layout=True,figsize=(18,6),ncols=4)
figHowTo,axsHowTo = plt.subplots(tight_layout=True,figsize=(14,6),ncols=3)
figHstr,axsHstr = plt.subplots(tight_layout=True,figsize=(8,6))
figOnlyLines,axsOnlyLines = plt.subplots(tight_layout=True,figsize=(12,6))
scan_dist = 33
# scale = 2.0970206260681152
scan_dist_forward = scan_dist + 0
scan_dist_backward = scan_dist_forward + 0

colors = cm.jet(np.linspace(0,1,len(BVal_holo)))

periodicity = []
periodicity_std = []
sigmas_m = []
sigmas_std = []
diff_m = []
diff_std = []
periodicity_m = []
periodicity_std = []
amps_m = []
amps_std = []
C_m = []
C_std = []
t_m = []
t_std = []
for i,p in enumerate(phi_holo):
    x = np.linspace(0,len(p),len(p))*scale
    axs[1].plot(x,p-p[0],color=colors[i])
    # p = phi_holo[i]
    bck = np.polyfit(x,p,2)
    
    axs[1].plot(x,np.polyval(bck,x)-np.polyval(bck,x)[0],color=colors[i],ls="dashed")
    # if i==0:
    #     axsHowTo[0].plot(x,p-p[0],color=colors[i],label="$\Delta \phi$")
    #     axsHowTo[0].plot(x,np.polyval(bck,x)-np.polyval(bck,x)[0],color="r",ls="dashed",label="Background")
        
    p = p - np.polyval(bck,x)
    
    N = 5
    # x = 
    p = np.convolve(p, np.ones(N)/N, mode='valid')
    x = np.convolve(x, np.ones(N)/N, mode='valid')
    
    # p = p - np.max(p) - 1e-50
    
    
    axs[0].plot(i+1,BVal_holo[i]*1e3,"o",color=colors[i])
    axsHstr.plot(i+1,BVal_holo[i]*1e3,"o",color=colors[i])
    if BVal_holo[i] >= 0:
        text_add1 = "+"
        if "rev" in files_holo[i]:
            text_add =  r'$\downarrow$'
        elif "rev" not in files_holo[i]:
            text_add = r'$\uparrow$'
    if BVal_holo[i] < 0:
        text_add1 = "-"
        if "rev" in files_holo[i]:
            text_add =  r'$\uparrow$'
        elif "rev" not in files_holo[i]:
            text_add =  r'$\downarrow$'
    axs[0].text((i+1)-1,BVal_holo[i]*1E3,text_add,color="k",fontsize=14,horizontalalignment = "left", verticalalignment = "center")
    axsHstr.text((i+1)-1,BVal_holo[i]*1E3,text_add,color="k",fontsize=14,horizontalalignment = "left", verticalalignment = "center")
    axs[2].plot(x,p+i*np.min(p),color=colors[i])
    # axsBckSub[1].
    peaks, peak_properties = find_peaks(p, prominence = 2)#,width=1)
    pi = 1/(p-np.max(p)-50e-9)
    valleys, _ = find_peaks(pi,prominence = 0.5)
    
    axs[2].plot(x[peaks],p[peaks]+i*np.min(p),"x",color=colors[i],alpha=0.4)
    axs[2].plot(x[valleys],p[valleys]+i*np.min(p),"o",color=colors[i],alpha=0.4)
    
    axsOnlyLines.plot(x,p,color=colors[i],label="Applied |H| = {:.0f} mT".format(BVal_holo[i]*1e3))
    axsOnlyLines.plot(x[peaks],p[peaks],"x",color=colors[i],alpha=0.4)
    axsOnlyLines.plot(x[valleys],p[valleys],"o",color=colors[i],alpha=0.4)
    
    diff_m.append(np.mean(p[peaks])-np.mean(p[valleys]))
    diff_std.append(np.mean([np.std(p[peaks]),np.std(p[valleys])]))
    
    if i==0:
        axsHowTo[0].plot(x,p,color=colors[i])
        # axsHowTo[1].annotate("",xy=(x[peaks[0]],p[peaks[0]]),xytext=(x[peaks[0]],p[valleys[1]]-p[peaks[0]]),
        #                      arrowprops=dict(arrowstyle="<->"))
        
        # plt.annotate("here", xy=(0, 0), xytext=(0, 2), arrowprops=dict(arrowstyle="->"))
        # axsHowTo[1].add_patch(arrow)
    
    peaks = peaks[peaks>scan_dist]
    valleys = valleys[valleys>scan_dist]
    sigmas = []
    mus = []
    amps = []
    Cvec = []
    tvec = []
    for j,pk in enumerate(peaks):
        xfit = x[pk-scan_dist_backward:pk+scan_dist_forward]
        
        pfit2 = p[pk-scan_dist_backward:pk+scan_dist_forward]
        # print(np.max(pfit2)-np.min(pfit2))

        sigma_fit_param = 20
        amp_fit_param = 5#np.max(pfit2)#+np.min(pfit2)
        
        p0 = [amp_fit_param,xfit[np.argmax(pfit2)],sigma_fit_param,2]#,2]

        p1,pcov = opt.curve_fit(CoshFunc,xfit,pfit2,p0=p0)
        residuals = pfit2- CoshFunc(xfit, *p1)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((pfit2-np.mean(pfit2))**2)
        r_squared = 1 - (ss_res / ss_tot)
        if r_squared > 0.9:
            fit_amp,fit_mu, fit_stdev,C = p1
            
            t = 2*np.pi*hbar*fit_amp/(e_const*fit_stdev*u0)
            
            axs[2].plot(xfit,CoshFunc(xfit,*p1)+i*np.min(p),color=colors[i],ls="dotted")
            axsOnlyLines.plot(xfit,CoshFunc(xfit,*p1),color=colors[i],ls="dotted")
            if i==0:
                axsHowTo[1].plot(xfit,pfit2,color="k",alpha=0.4,lw=3)
                axsHowTo[1].plot(xfit,CoshFunc(xfit,*p1),color="b",ls="dotted")
                
                axsHowTo[0].plot(fit_mu,p[pk]+i*np.min(p),"x",color=colors[i],alpha=0.4)#,label="Peaks")
                # axsHowTo[1].plot(x[valleys],p[valleys]+i*np.min(p),"o",color=colors[i],alpha=0.4,label="Valleys")
                if j==0:
                    axsHowTo[1].plot([],[],color="b",ls="dotted",label="$\Delta \phi_{theo,peaks}$")
                    axsHowTo[0].plot([],[],color=colors[i],marker="x",ls="",label="Peaks")
            sigmas.append(fit_stdev)
            mus.append(fit_mu)
            amps.append(-fit_amp)
            Cvec.append(C)
            tvec.append(t)
    sigmas_pk = np.mean(sigmas)
    sigmas_pk_std = np.std(sigmas)
    amps_pk = np.mean(amps)
    amps_pk_std = np.mean(amps)
    
    t_pk = np.mean(tvec)
    t_pk_std = np.std(tvec)
    C_pk = np.mean(Cvec)
    C_pk_std = np.mean(Cvec)
    sigmas = []
    mus = []
    amps = []
    Cvec = []
    tvec = []
    for j,pk in enumerate(valleys):
        xfit = x[pk-scan_dist_backward:pk+scan_dist_forward]
        
        pfit2 = p[pk-scan_dist_backward:pk+scan_dist_forward]

        sigma_fit_param = 20
        amp_fit_param = 5#np.max(pfit2)#+np.min(pfit2)
        
        p0 = [amp_fit_param,xfit[np.argmax(pfit2)],sigma_fit_param,np.max(pfit2)]#,2]

        p1,pcov = opt.curve_fit(CoshFunc,xfit,pfit2,p0=p0)
        residuals = pfit2- CoshFunc(xfit, *p1)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((pfit2-np.mean(pfit2))**2)
        r_squared = 1 - (ss_res / ss_tot)
        if r_squared > 0.9:
            fit_amp,fit_mu, fit_stdev,C = p1
            t = (2*np.pi*hbar*fit_amp)/(e_const*fit_stdev*u0)
            axs[2].plot(xfit,CoshFunc(xfit,*p1)+i*np.min(p),color=colors[i],ls="dashed")
            axsOnlyLines.plot(xfit,CoshFunc(xfit,*p1),color=colors[i],ls="dashed")
            if i==0:
                axsHowTo[1].plot(xfit,CoshFunc(xfit,*p1),color="r",ls="dashed")
                axsHowTo[0].plot(fit_mu,p[pk]+i*np.min(p),"o",color=colors[i],alpha=0.4)
                if j==0:
                    axsHowTo[1].plot([],[],color="r",ls="dashed",label="$\Delta \phi_{theo,valleys}$")
                    axsHowTo[0].plot([],[],color=colors[i],marker="o",ls="",label="Valleys")
            sigmas.append(fit_stdev)
            mus.append(fit_mu)
            amps.append(fit_amp)
            Cvec.append(C)
            tvec.append(t)
    sigmas_val = np.mean(sigmas)
    sigmas_val_std = np.std(sigmas)
    amps_val = np.mean(amps)
    amps_val_std = np.std(amps)
    C_val = np.mean(Cvec)
    C_val_std = np.mean(Cvec)
    
    t_val = np.mean(tvec)
    t_val_std = np.std(tvec)
    
    sigmas_i = sigmas_pk#(sigmas_pk+sigmas_val)/2
    sigmas_i_std = sigmas_pk_std#(sigmas_pk_std+sigmas_val_std)/2
    
    amps_i = (amps_pk+amps_val)/2
    amps_i_std = (amps_pk_std+amps_val_std)/2
    
    C_i = (C_pk + C_val)/2
    C_i_std = (C_pk_std+C_val_std)/2
    
    t_i = (t_pk+t_val)/2
    t_i_std = (t_pk_std+t_val_std)/2
    
    sigmas_m.append(sigmas_i)
    sigmas_std.append(sigmas_i_std)
    amps_m.append(amps_i)
    amps_std.append(amps_i_std)
    C_m.append(C_i)
    C_std.append(C_i_std)    
    t_m.append(t_i)
    t_std.append(t_i_std)
    
    diff_mus = [mus[i+1]-mus[i] for i in range(len(mus)-1)]
    periodicity_m.append(np.mean(diff_mus))
    periodicity_std.append(np.std(diff_mus))
    
    FT = np.fft.fftshift(np.fft.fft(p))#[:nfft//2]
    x_interval = x[1]-x[0]
    nfft = len(p)
    # FT[0] = 0
    freqs = np.arange(1/x_interval/-2,1/x_interval/2,1/x_interval/nfft)
    
    FT_norm_abs = abs(FT)/np.max(abs(FT))
    
    FT_Halfed = FT_norm_abs[nfft//2:]
    freqs_Halfed = freqs[nfft//2:]
    idx_max = np.argmax(FT_Halfed[1:])
    idx_max = idx_max+1
    FT_Max = freqs_Halfed[idx_max]
    if i==0:
        figFFT,axsFFT = plt.subplots(tight_layout=True)
        axsFFT.plot(freqs,FT_norm_abs,color=colors[i])
        axsFFT.plot(freqs_Halfed[idx_max],FT_Halfed[idx_max],"rx",ms=10)
    if i==4:
        axsFFT.plot(freqs,FT_norm_abs,color=colors[i])
        axsFFT.plot(freqs_Halfed[idx_max],FT_Halfed[idx_max],"rx",ms=10)
    if float(BVal_holo[i]*1e3) == float(120):
        # figFFT,axsFFT = plt.subplots(tight_layout=True)
        axsFFT.plot(freqs,FT_norm_abs,color="r",lw=2)
        axsFFT.plot(freqs_Halfed[idx_max],FT_Halfed[idx_max],"rx",ms=10)
    # print(FT_Max)
    
    if i==0:
        axsHowTo[2].plot(freqs_Halfed,FT_Halfed,color=colors[i],label="$FFT(\Delta \phi)$")
        axsHowTo[2].plot(freqs[int(nfft//2 + idx_max)],FT_norm_abs[int(nfft//2 + idx_max)],"x",color="r",ms=10,label="$max(FFT(\Delta \phi))$")

    diffs_mus = []
    for j in range(len(mus)-1):
        diffs_mus.append(mus[j+1]-mus[j])
    # diffs
    # periodicity.append(np.mean(diffs_mus))
    periodicity.append(1/FT_Max)
    periodicity_std.append(np.std(diffs_mus))

periodicity = [np.mean(periodicity) for i in range(len(periodicity))]
periodicity = np.asarray(periodicity)

axs[1].set_xlabel("Distance [nm]",fontsize=20)
axs[1].set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
axs[0].set_ylabel("Applied |H| [mT]",fontsize=20)
axs[0].set_xlabel("Measurement #",fontsize=20)
axs[2].set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
axs[2].set_xlabel("Distance [nm]",fontsize=20)
axs[3].set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
# Another way of removing MIP
dphi_holo_20mT = 0.5*(phi_holo[1] - phi_holo[11])
dphi_holo_40mT = 0.5*(phi_holo[2] - phi_holo[12])
dphi_holo_0mT = 0.5*(phi_holo[0] - phi_holo[10])
dphi_holo_120mT = 0.5*(phi_holo[6] - phi_holo[16])
# axs[3].plot(x,dphi_holo_0mT-dphi_holo_0mT[0],color="k",lw="2",label="0 mT")
# axs[3].plot(x,dphi_holo_20mT-dphi_holo_20mT[0],color="b",lw="2",label="20 mT")
# # axs[3].plot(x,dphi_holo_40mT-dphi_holo_40mT[0],color="r",lw="2",label="40 mT")
# # axs[3].plot(x,dphi_holo_120mT-dphi_holo_120mT[0],color="m",lw="2",label="120 mT")
# axs[3].legend(loc="upper left",fontsize=16)
# for n in range(4):
#     axs[n].tick_params(axis="both",labelsize=14)
#     axs[n].text(-0.1,0.99,lbls[n],horizontalalignment="right",verticalalignment="top",transform=axs[n].transAxes,fontsize=30,fontname="Arial")
#     if n>0:
#         axs[n].set_xlim([x[0],x[-1]])
if save_figs:
    fig.savefig(save_path+"\\"+"OA_AllAppliedH_WithBckSub_withnewmethod_only20mT" + save_add + ".png",bbox_inches="tight")
#####
for n,ax in enumerate(axsHowTo.flat):
    # if n<3:
    #     if n==0:
    #         ax0 = ax
    #     ax.sharey(ax0)
    axsHowTo[n].tick_params(axis="both",labelsize=14)
    axsHowTo[n].legend(loc="upper right",fontsize=16)
    axsHowTo[n].text(-0.15,0.95,lbls[n],horizontalalignment="right",verticalalignment="top",transform=axsHowTo[n].transAxes,fontsize=30,fontname="Arial")
    if n<range(3)[-1]:
        
        axsHowTo[n].set_xlabel("Distance [nm]",fontsize=20)
        
        axsHowTo[n].set_xlim([x[0],x[-1]])
    else:
        axsHowTo[n].set_xlim([freqs_Halfed[0],freqs_Halfed[-1]])


axsHowTo[0].set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
axsHowTo[2].set_ylabel("Normalized Intensity",fontsize=20)
axsHowTo[2].set_xlabel("Frequency [1/nm]",fontsize=20)
if save_figs:
    figHowTo.savefig(save_path+"\\"+"OA_HowTo" + save_add + ".png",bbox_inches="tight")

##############
axsHstr.tick_params(axis="both",labelsize=14)
axsHstr.set_ylabel("Applied |H| [mT]",fontsize=20)
axsHstr.set_xlabel("Measurement #",fontsize=20)

if save_figs:
    figHstr.savefig(save_path+"\\"+"AppliedH" + save_add + ".png",bbox_inches="tight")

####¤####################
axsOnlyLines.set_xlabel("Distance [nm]",fontsize=20)
axsOnlyLines.set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
axsOnlyLines.tick_params(axis="both",labelsize=14)
# axsOnlyLines.legend(loc="upper right",fontsize=16)
axsOnlyLines.set_xlim([x[0],x[-1]])
if save_figs:
    figOnlyLines.savefig(save_path+"\\"+"AllProfiles_OnlyProfiles_WithBckSub" + save_add + ".png",bbox_inches="tight")
################################
t_m = np.asarray(t_m)
t_std = np.asarray(t_std)
fig,axs = plt.subplots(tight_layout=True,nrows=4,sharex=True,figsize=(10,8))
# axs.errorbar(BVal_holo,diff_m,yerr=diff_std,marker="o",ls="solid",color="k",capsize=10)
for i in range(len(BVal_holo)):
    
    axs[0].errorbar(BVal_holo[i]*1e3,sigmas_m[i],yerr=sigmas_std[i],marker="o",color=colors[i],ls="solid",capsize=10)
    # axs[1,0].plot(BVal_holo[i],diff_m[i],marker="o",ls="solid",color=colors[i])
    axs[1].errorbar(BVal_holo[i]*1e3,amps_m[i],yerr=amps_std[i],marker="o",ls="solid",color=colors[i],capsize=10)
    # axs[1,1].errorbar(BVal_holo[i],periodicity[i],yerr=periodicity_std[i],marker="o",color=colors[i],capsize=10)
    axs[2].errorbar(BVal_holo[i]*1e3,periodicity[i],color=colors[i],marker="o")
    
    
    axs[3].errorbar(BVal_holo[i]*1e3,t_m[i]*1e9,yerr=t_std[i]*1e9,marker="o",ls="solid",color=colors[i],capsize=10)

axs[0].plot(BVal_holo*1e3,sigmas_m,"k")
axs[1].plot(BVal_holo*1e3,amps_m,"k")
axs[2].plot(BVal_holo*1e3,periodicity,"k")
axs[3].plot(BVal_holo*1e3,t_m*1e9,"k")
for n in range(4):
    axs[n].tick_params(axis="both",labelsize=14)
    
    axs[n].text(-0.05,0.99,lbls[n],horizontalalignment="right",verticalalignment="top",transform=axs[n].transAxes,fontsize=30,fontname="Arial")

axs[-1].set_xlabel("Applied |H| [mT]",fontsize=20)
axs[0].set_ylabel("$\sigma  \ [nm]$",fontsize=20)
axs[1].set_ylabel("A [rad]",fontsize=20)
axs[2].set_ylabel("$\lambda \ [nm]$",fontsize=20)
axs[3].set_ylabel("$Thickness \ [nm]$",fontsize=20)
if save_figs:
    fig.savefig(save_path+"\\"+"OA_AllAppliedH_FittedParameters_witht" + save_add + ".png",bbox_inches="tight")
fig,axs = plt.subplots(tight_layout=True,nrows=4,sharex=True,figsize=(10,8))
# axs.errorbar(BVal_holo,diff_m,yerr=diff_std,marker="o",ls="solid",color="k",capsize=10)
for i in range(len(BVal_holo)):
    
    axs[0].errorbar(BVal_holo[i]*1e3,sigmas_m[i],yerr=sigmas_std[i],marker="o",color=colors[i],ls="solid",capsize=10)
    # axs[1,0].plot(BVal_holo[i],diff_m[i],marker="o",ls="solid",color=colors[i])
    axs[1].errorbar(BVal_holo[i]*1e3,amps_m[i],yerr=amps_std[i],marker="o",ls="solid",color=colors[i],capsize=10)
    # axs[1,1].errorbar(BVal_holo[i],periodicity[i],yerr=periodicity_std[i],marker="o",color=colors[i],capsize=10)
    axs[2].errorbar(BVal_holo[i]*1e3,periodicity[i],color=colors[i],marker="o")
    
    axs[3].errorbar(BVal_holo[i]*1e3,C_m[i],yerr=C_std[i],marker="o",ls="solid",color=colors[i],capsize=10)
    
axs[0].plot(BVal_holo*1e3,sigmas_m,"k")
axs[1].plot(BVal_holo*1e3,amps_m,"k")
axs[2].plot(BVal_holo*1e3,periodicity,"k")

axs[3].plot(BVal_holo*1e3,C_m,"k")
for n in range(4):
    axs[n].tick_params(axis="both",labelsize=14)
    
    axs[n].text(-0.05,0.99,lbls[n],horizontalalignment="right",verticalalignment="top",transform=axs[n].transAxes,fontsize=30,fontname="Arial")

axs[-1].set_xlabel("Applied |H| [mT]",fontsize=20)
axs[0].set_ylabel("$\sigma  \ [nm]$",fontsize=20)
axs[1].set_ylabel("A [rad]",fontsize=20)
axs[2].set_ylabel("$\lambda \ [nm]$",fontsize=20)
axs[3].set_ylabel("Offset [rad]",fontsize=20)

#%%
plt.close("all")
dphi_holo_20mT = 0.5*(phi_holo[1] - phi_holo[11])
dphi_holo_40mT = 0.5*(phi_holo[2] - phi_holo[12])
dphi_holo_0mT = 0.5*(phi_holo[0] - phi_holo[10])
dphi_holo_120mT = 0.5*(phi_holo[6] - phi_holo[16])
peaks, peak_properties = find_peaks(dphi_holo_20mT, prominence = 2)
amps = []
mus = []
sigmas = []
Cvec = []
for j,pk in enumerate(peaks):
        xfit = x[pk-scan_dist_backward:pk+scan_dist_forward]
        
        pfit2 = p[pk-scan_dist_backward:pk+scan_dist_forward]

        sigma_fit_param = 20
        amp_fit_param = 1#np.max(pfit2)#+np.min(pfit2)
        
        p0 = [amp_fit_param,xfit[np.argmax(pfit2)],sigma_fit_param,np.max(pfit2)]#,2]

        p1,pcov = opt.curve_fit(CoshFunc,xfit,pfit2,p0=p0)
        residuals = pfit2- CoshFunc(xfit, *p1)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((pfit2-np.mean(pfit2))**2)
        r_squared = 1 - (ss_res / ss_tot)
        if r_squared > 0.9:
            fit_amp,fit_mu, fit_stdev,C = p1
            
            sigmas.append(fit_stdev)
            mus.append(fit_mu)
            amps.append(-fit_amp)
            Cvec.append(C)

mean_sigma = np.mean(sigmas)
mean_amp = np.mean(amps)
fig,axs = plt.subplots(tight_layout=True)
axs.plot(x,dphi_holo_0mT-dphi_holo_0mT[0],"k",lw="2",label="0 mT")
axs.plot(x,dphi_holo_20mT-dphi_holo_20mT[0],"b",lw="2",label="20 mT")
axs.plot(x,dphi_holo_40mT-dphi_holo_40mT[0],"r",lw="2",label="40 mT")

axs.set_xlabel("Distance [nm]",fontsize=20)
axs.set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
axs.tick_params(axis="both",labelsize=14)
axs.legend(loc="upper left",fontsize=16)

#%%
def CoshFunc(x,A,x0,sigma,C):
    return A*np.log( np.cosh( (x-x0) / sigma ) ) + C

def SqrtFunc(x,A,x0,C):
    return 1/np.sqrt(1 - (x-x0)) + C
########################################
save_figs = 0
########################################
e_const = 1.602e-19
hbar = 1.05e-34
u0 = 1.256*1e-6
M0 = 2 #T
B0 = u0*M0
plt.close("all")
lbls = ["(a)","(b)","(c)","(d)","(e)"]
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images'

if "Location_C" in path_holo:
    save_add = "_locC"
if "Location_1" in path_holo:
    save_add = "_loc1"
fig,axs = plt.subplots(tight_layout=True,figsize=(18,6),ncols=4)
figHowTo,axsHowTo = plt.subplots(tight_layout=True,figsize=(14,6),ncols=3)
figHstr,axsHstr = plt.subplots(tight_layout=True,figsize=(8,6))
figOnlyLines,axsOnlyLines = plt.subplots(tight_layout=True,figsize=(12,6))
scan_dist = 35
# scale = 2.0970206260681152
scan_dist_forward = scan_dist + 0
scan_dist_backward = scan_dist_forward + 0

colors = cm.jet(np.linspace(0,1,len(BVal_holo)))

periodicity_pk = []
periodicity_pk_std = []
periodicity_valley = []
periodicity_valley_std = []
sigmas_pk_vec = []
sigmas_pk_std_vec = []
sigmas_valley_vec = []
sigmas_valley_std_vec = []
sigmas_m = []
sigmas_std = []
diff_m = []
diff_std = []
periodicity_m = []
periodicity_std = []
amps_m = []
amps_std = []
amps_pk_vec = []
amps_pk_std_vec = []
amps_valley_vec = []
amps_valley_std_vec = []
C_m = []
C_std = []
t_m = []
t_std = []
for i,p in enumerate(phi_holo):
    x = np.linspace(0,len(p),len(p))*scale
    axs[1].plot(x,p-p[0],color=colors[i])
    # p = phi_holo[i]
    bck = np.polyfit(x,p,2)
    
    axs[1].plot(x,np.polyval(bck,x)-np.polyval(bck,x)[0],color=colors[i],ls="dashed")
    # if i==0:
    #     axsHowTo[0].plot(x,p-p[0],color=colors[i],label="$\Delta \phi$")
    #     axsHowTo[0].plot(x,np.polyval(bck,x)-np.polyval(bck,x)[0],color="r",ls="dashed",label="Background")
        
    p = p - np.polyval(bck,x)
    
    N = 5
    # x = 
    p = np.convolve(p, np.ones(N)/N, mode='valid')
    x = np.convolve(x, np.ones(N)/N, mode='valid')
    
    # p = p - np.max(p) - 1e-50
    
    
    axs[0].plot(i+1,BVal_holo[i]*1e3,"o",color=colors[i])
    axsHstr.plot(i+1,BVal_holo[i]*1e3,"o",color=colors[i])
    if BVal_holo[i] > 0:
        text_add1 = "+"
        if "rev" in files_holo[i]:
            text_add =  r'$\downarrow$'
        elif "rev" not in files_holo[i]:
            text_add = r'$\uparrow$'
    if BVal_holo[i] < 0:
        text_add1 = "-"
        if "rev" in files_holo[i]:
            text_add =  r'$\uparrow$'
        elif "rev" not in files_holo[i]:
            text_add =  r'$\downarrow$'
    axs[0].text((i+1)-1,BVal_holo[i]*1E3,text_add,color="k",fontsize=14,horizontalalignment = "left", verticalalignment = "center")
    axsHstr.text((i+1)-1,BVal_holo[i]*1E3,text_add,color="k",fontsize=14,horizontalalignment = "left", verticalalignment = "center")
    axs[2].plot(x,p+i*np.min(p),color=colors[i])
    # axsBckSub[1].
    peaks, peak_properties = find_peaks(p, prominence = 2)#,width=1)
    pi = 1/(p-np.max(p)-50e-9)
    valleys, _ = find_peaks(pi,prominence = 0.5)
    
    axs[2].plot(x[peaks],p[peaks]+i*np.min(p),"x",color=colors[i],alpha=0.4)
    axs[2].plot(x[valleys],p[valleys]+i*np.min(p),"o",color=colors[i],alpha=0.4)
    
    axsOnlyLines.plot(x,p,color=colors[i],label="Applied |H| = {:.0f} mT".format(BVal_holo[i]*1e3))
    axsOnlyLines.plot(x[peaks],p[peaks],"x",color=colors[i],alpha=0.4)
    axsOnlyLines.plot(x[valleys],p[valleys],"o",color=colors[i],alpha=0.4)
    
    diff_m.append(np.mean(p[peaks])-np.mean(p[valleys]))
    diff_std.append(np.mean([np.std(p[peaks]),np.std(p[valleys])]))
    
    if i==0:
        axsHowTo[0].plot(x,p,color=colors[i])
        # axsHowTo[1].annotate("",xy=(x[peaks[0]],p[peaks[0]]),xytext=(x[peaks[0]],p[valleys[1]]-p[peaks[0]]),
        #                      arrowprops=dict(arrowstyle="<->"))
        
        # plt.annotate("here", xy=(0, 0), xytext=(0, 2), arrowprops=dict(arrowstyle="->"))
        # axsHowTo[1].add_patch(arrow)
    
    peaks = peaks[peaks>scan_dist]
    valleys = valleys[valleys>scan_dist]
    sigmas = []
    mus = []
    amps = []
    Cvec = []
    tvec = []
    for j,pk in enumerate(peaks):
        xfit = x[pk-scan_dist_backward:pk+scan_dist_forward]
        
        pfit2 = p[pk-scan_dist_backward:pk+scan_dist_forward]
        # print(np.max(pfit2)-np.min(pfit2))

        sigma_fit_param = 20
        amp_fit_param = 5#np.max(pfit2)#+np.min(pfit2)
        
        p0 = [amp_fit_param,xfit[np.argmax(pfit2)],sigma_fit_param,2]#,2]

        p1,pcov = opt.curve_fit(CoshFunc,xfit,pfit2,p0=p0)#,bounds=([-np.inf,-np.inf,0,-np.inf],[np.inf,np.inf,1e3,np.inf]))#,(-1e9,1e9),(0,1e9),(-1e9,1e9)))
        residuals = pfit2- CoshFunc(xfit, *p1)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((pfit2-np.mean(pfit2))**2)
        r_squared = 1 - (ss_res / ss_tot)
        if r_squared > 0.75:
            if p1[2] > 0:
                fit_amp,fit_mu, fit_stdev,C = p1
                
                # if fit_stdev < 0:
                #     fit_stdev = fit_stdev*-1
                
                t = 2*np.pi*hbar*fit_amp/(e_const*fit_stdev*u0)
                
                axs[2].plot(xfit,CoshFunc(xfit,*p1)+i*np.min(p),color=colors[i],ls="dotted")
                axsOnlyLines.plot(xfit,CoshFunc(xfit,*p1),color=colors[i],ls="dotted")
                if i==0:
                    axsHowTo[1].plot(xfit,pfit2,color="k",alpha=0.4,lw=3)
                    axsHowTo[1].plot(xfit,CoshFunc(xfit,*p1),color="b",ls="dotted")
                    
                    axsHowTo[0].plot(fit_mu,p[pk]+i*np.min(p),"x",color=colors[i],alpha=0.4)#,label="Peaks")
                    # axsHowTo[1].plot(x[valleys],p[valleys]+i*np.min(p),"o",color=colors[i],alpha=0.4,label="Valleys")
                    if j==0:
                        axsHowTo[1].plot([],[],color="b",ls="dotted",label="$\Delta \phi_{theo,peaks}$")
                        axsHowTo[0].plot([],[],color=colors[i],marker="x",ls="",label="Peaks")
                sigmas.append(fit_stdev)
                # print(sigmas)
                mus.append(fit_mu)
                amps.append(-fit_amp)
                Cvec.append(C)
                tvec.append(t)
                print(len(sigmas))
    sigmas_pk = np.mean(sigmas)
    # print(np.mean(sigmas))
    sigmas_pk_std = np.std(sigmas)
    amps_pk = np.mean(amps)
    amps_pk_std = np.mean(amps)
    
    sigmas_pk_vec.append(sigmas_pk)
    sigmas_pk_std_vec.append(sigmas_pk_std)
    amps_pk_vec.append(amps_pk)
    amps_pk_std_vec.append(amps_pk_std)
    
    t_pk = np.mean(tvec)
    t_pk_std = np.std(tvec)
    C_pk = np.mean(Cvec)
    C_pk_std = np.mean(Cvec)
    sigmas = []
    mus = []
    amps = []
    Cvec = []
    tvec = []
    for j,pk in enumerate(valleys):
        xfit = x[pk-scan_dist_backward:pk+scan_dist_forward]
        
        pfit2 = p[pk-scan_dist_backward:pk+scan_dist_forward]

        sigma_fit_param = 20
        amp_fit_param = 5#np.max(pfit2)#+np.min(pfit2)
        
        p0 = [amp_fit_param,xfit[np.argmax(pfit2)],sigma_fit_param,np.max(pfit2)]#,2]

        p1,pcov = opt.curve_fit(CoshFunc,xfit,pfit2,p0=p0)
        residuals = pfit2- CoshFunc(xfit, *p1)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((pfit2-np.mean(pfit2))**2)
        r_squared = 1 - (ss_res / ss_tot)
        if r_squared > 0.9:
            fit_amp,fit_mu, fit_stdev,C = p1
            
            if fit_stdev < 0:
                fit_stdev = fit_stdev * -1
            t = (2*np.pi*hbar*fit_amp)/(e_const*fit_stdev*u0)
            axs[2].plot(xfit,CoshFunc(xfit,*p1)+i*np.min(p),color=colors[i],ls="dashed")
            axsOnlyLines.plot(xfit,CoshFunc(xfit,*p1),color=colors[i],ls="dashed")
            if i==0:
                axsHowTo[1].plot(xfit,CoshFunc(xfit,*p1),color="r",ls="dashed")
                axsHowTo[0].plot(fit_mu,p[pk]+i*np.min(p),"o",color=colors[i],alpha=0.4)
                if j==0:
                    axsHowTo[1].plot([],[],color="r",ls="dashed",label="$\Delta \phi_{theo,valleys}$")
                    axsHowTo[0].plot([],[],color=colors[i],marker="o",ls="",label="Valleys")
            sigmas.append(fit_stdev)
            mus.append(fit_mu)
            amps.append(fit_amp)
            Cvec.append(C)
            tvec.append(t)
    sigmas_val = np.mean(sigmas)
    sigmas_val_std = np.std(sigmas)
    amps_val = np.mean(amps)
    amps_val_std = np.std(amps)
    
    sigmas_valley_vec.append(sigmas_val)
    sigmas_valley_std_vec.append(sigmas_val_std)
    amps_valley_vec.append(amps_val)
    amps_valley_std_vec.append(amps_val_std)
    
    diff_mus = [mus[i+1]-mus[i] for i in range(len(mus)-1)]
    periodicity_m.append(np.mean(diff_mus))
    periodicity_std.append(np.std(diff_mus))
    
    FT = np.fft.fftshift(np.fft.fft(p))#[:nfft//2]
    x_interval = x[1]-x[0]
    nfft = len(p)
    # FT[0] = 0
    freqs = np.arange(1/x_interval/-2,1/x_interval/2,1/x_interval/nfft)
    
    FT_norm_abs = abs(FT)/np.max(abs(FT))
    
    FT_Halfed = FT_norm_abs[nfft//2:]
    freqs_Halfed = freqs[nfft//2:]
    idx_max = np.argmax(FT_Halfed[1:])
    idx_max = idx_max+1
    FT_Max = freqs_Halfed[idx_max]
    if i==0:
        figFFT,axsFFT = plt.subplots(tight_layout=True)
        axsFFT.plot(freqs,FT_norm_abs,color=colors[i])
        axsFFT.plot(freqs_Halfed[idx_max],FT_Halfed[idx_max],"rx",ms=10)
    if i==4:
        axsFFT.plot(freqs,FT_norm_abs,color=colors[i])
        axsFFT.plot(freqs_Halfed[idx_max],FT_Halfed[idx_max],"rx",ms=10)
    if float(BVal_holo[i]*1e3) == float(120):
        # figFFT,axsFFT = plt.subplots(tight_layout=True)
        axsFFT.plot(freqs,FT_norm_abs,color="r",lw=2)
        axsFFT.plot(freqs_Halfed[idx_max],FT_Halfed[idx_max],"rx",ms=10)
    # print(FT_Max)
    
    if i==0:
        axsHowTo[2].plot(freqs_Halfed,FT_Halfed,color=colors[i],label="$FFT(\Delta \phi)$")
        axsHowTo[2].plot(freqs[int(nfft//2 + idx_max)],FT_norm_abs[int(nfft//2 + idx_max)],"x",color="r",ms=10,label="$max(FFT(\Delta \phi))$")

    diffs_mus = []
    for j in range(len(mus)-1):
        diffs_mus.append(mus[j+1]-mus[j])
    # diffs
    # periodicity.append(np.mean(diffs_mus))
    periodicity_pk.append(1/FT_Max)
    periodicity_pk_std.append(np.std(diffs_mus))
 

periodicity_pk = [np.mean(periodicity_pk) for i in range(len(periodicity_pk))]
periodicity_pk = np.asarray(periodicity_pk)
axs[1].set_xlabel("Distance [nm]",fontsize=20)
axs[1].set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
axs[0].set_ylabel("Applied |H| [mT]",fontsize=20)
axs[0].set_xlabel("Measurement #",fontsize=20)
axs[2].set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
axs[2].set_xlabel("Distance [nm]",fontsize=20)
axs[3].set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
# Another way of removing MIP
dphi_holo_20mT = 0.5*(phi_holo[1] - phi_holo[11])
dphi_holo_40mT = 0.5*(phi_holo[2] - phi_holo[12])
dphi_holo_0mT = 0.5*(phi_holo[0] - phi_holo[10])
dphi_holo_120mT = 0.5*(phi_holo[6] - phi_holo[16])
# axs[3].plot(x,dphi_holo_0mT-dphi_holo_0mT[0],color="k",lw="2",label="0 mT")
# axs[3].plot(x,dphi_holo_20mT-dphi_holo_20mT[0],color="b",lw="2",label="20 mT")
# # axs[3].plot(x,dphi_holo_40mT-dphi_holo_40mT[0],color="r",lw="2",label="40 mT")
# # axs[3].plot(x,dphi_holo_120mT-dphi_holo_120mT[0],color="m",lw="2",label="120 mT")
# axs[3].legend(loc="upper left",fontsize=16)
# for n in range(4):
#     axs[n].tick_params(axis="both",labelsize=14)
#     axs[n].text(-0.1,0.99,lbls[n],horizontalalignment="right",verticalalignment="top",transform=axs[n].transAxes,fontsize=30,fontname="Arial")
#     if n>0:
#         axs[n].set_xlim([x[0],x[-1]])
if save_figs:
    fig.savefig(save_path+"\\"+"OA_AllAppliedH_WithBckSub_withnewmethod_only20mT" + save_add + ".png",bbox_inches="tight")
#####
for n,ax in enumerate(axsHowTo.flat):
    # if n<3:
    #     if n==0:
    #         ax0 = ax
    #     ax.sharey(ax0)
    axsHowTo[n].tick_params(axis="both",labelsize=14)
    axsHowTo[n].legend(loc="upper right",fontsize=16)
    axsHowTo[n].text(-0.15,0.95,lbls[n],horizontalalignment="right",verticalalignment="top",transform=axsHowTo[n].transAxes,fontsize=30,fontname="Arial")
    if n<range(3)[-1]:
        
        axsHowTo[n].set_xlabel("Distance [nm]",fontsize=20)
        
        axsHowTo[n].set_xlim([x[0],x[-1]])
    else:
        axsHowTo[n].set_xlim([freqs_Halfed[0],freqs_Halfed[-1]])


axsHowTo[0].set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
axsHowTo[2].set_ylabel("Normalized Intensity",fontsize=20)
axsHowTo[2].set_xlabel("Frequency [1/nm]",fontsize=20)
if save_figs:
    figHowTo.savefig(save_path+"\\"+"OA_HowTo" + save_add + ".png",bbox_inches="tight")

##############
axsHstr.tick_params(axis="both",labelsize=14)
axsHstr.set_ylabel("Applied |H| [mT]",fontsize=20)
axsHstr.set_xlabel("Measurement #",fontsize=20)
if save_figs:
    figHstr.savefig(save_path+"\\"+"AppliedH" + save_add + ".png",bbox_inches="tight")

####¤####################
axsOnlyLines.set_xlabel("Distance [nm]",fontsize=20)
axsOnlyLines.set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
axsOnlyLines.tick_params(axis="both",labelsize=14)
# axsOnlyLines.legend(loc="upper right",fontsize=16)
axsOnlyLines.set_xlim([x[0],x[-1]])
if save_figs:
    figOnlyLines.savefig(save_path+"\\"+"AllProfiles_OnlyProfiles_WithBckSub" + save_add + ".png",bbox_inches="tight")
################################
t_m = np.asarray(t_m)
t_std = np.asarray(t_std)



fig,axs = plt.subplots(tight_layout=True,nrows=3,sharex=True,figsize=(10,8))
idx_list_sorted = np.argsort(BVal_holo)
BVal_holo_sorted = []
sigmas_pk_vec_sorted = []
for i in idx_list_sorted:
    sigmas_pk_vec_sorted.append(sigmas_pk_vec[i])
    BVal_holo_sorted.append(BVal_holo[i])
BVal_holo_sorted = np.asarray(BVal_holo_sorted)
# axs.errorbar(BVal_holo,diff_m,yerr=diff_std,marker="o",ls="solid",color="k",capsize=10)
for i in range(len(BVal_holo)):
    
    axs[0].errorbar(BVal_holo[i]*1e3,sigmas_pk_vec[i],yerr=sigmas_pk_std_vec[i],marker="o",color=colors[i],ls="solid",capsize=10)
    # axs[1,0].plot(BVal_holo[i],diff_m[i],marker="o",ls="solid",color=colors[i])
    axs[1].errorbar(BVal_holo[i]*1e3,amps_pk_vec[i],yerr=amps_pk_std_vec[i],marker="o",ls="solid",color=colors[i],capsize=10)
    # axs[1,1].errorbar(BVal_holo[i],periodicity[i],yerr=periodicity_std[i],marker="o",color=colors[i],capsize=10)
    axs[2].errorbar(BVal_holo[i]*1e3,periodicity_pk[i],color=colors[i],marker="o")

axs[0].plot(BVal_holo*1e3,sigmas_pk_vec,"k",label="Experimental")
p0 = [1,0,10]
theo_fit,pcov = opt.curve_fit(SqrtFunc,BVal_holo_sorted,sigmas_pk_vec_sorted)#,p0=p0)
# fig,axs = plt.subplots(tight_layout=True)
BVal_holo_fit = np.linspace(BVal_holo_sorted[0],BVal_holo_sorted[-1],100)
axs[0].plot(BVal_holo_fit*1e3,SqrtFunc(BVal_holo_fit,*theo_fit),"r--",label="Theory")
axs[0].legend(loc="upper left",fontsize=14)
axs[1].plot(BVal_holo*1e3,amps_pk_vec,"k")
axs[2].plot(BVal_holo*1e3,periodicity_pk,"k")
for n in range(3):
    axs[n].tick_params(axis="both",labelsize=14)
    
    axs[n].text(-0.05,0.99,lbls[n],horizontalalignment="right",verticalalignment="top",transform=axs[n].transAxes,fontsize=30,fontname="Arial")

axs[-1].set_xlabel("Applied |H| [mT]",fontsize=20)
axs[0].set_ylabel("$\sigma  \ [nm]$",fontsize=20)
axs[1].set_ylabel("A [rad]",fontsize=20)
axs[2].set_ylabel("$\lambda \ [nm]$",fontsize=20)
if save_figs:
    fig.savefig(save_path+"\\"+"OA_AllAppliedH_FittedParameters_OnlyPeaks" + save_add + ".png",bbox_inches="tight")

valley_pks_titles = ["Peaks","Valleys"]
fig,axs = plt.subplots(tight_layout=True,nrows=2,ncols=2,sharex=True,sharey=True,figsize=(10,8))
# axs.errorbar(BVal_holo,diff_m,yerr=diff_std,marker="o",ls="solid",color="k",capsize=10)
for i in range(len(BVal_holo)):
    
    axs[0,0].errorbar(BVal_holo[i]*1e3,sigmas_pk_vec[i],yerr=sigmas_pk_std_vec[i],marker="o",color=colors[i],ls="solid",capsize=10)
    # axs[1,0].plot(BVal_holo[i],diff_m[i],marker="o",ls="solid",color=colors[i])
    axs[1,0].errorbar(BVal_holo[i]*1e3,amps_pk_vec[i],yerr=amps_pk_std_vec[i],marker="o",ls="solid",color=colors[i],capsize=10)
    # axs[1,1].errorbar(BVal_holo[i],periodicity[i],yerr=periodicity_std[i],marker="o",color=colors[i],capsize=10)
    # axs[2,0].errorbar(BVal_holo[i]*1e3,periodicity_pk[i],color=colors[i],marker="o")
    
    axs[0,1].errorbar(BVal_holo[i]*1e3,sigmas_valley_vec[i],yerr=sigmas_valley_std_vec[i],marker="o",color=colors[i],ls="solid",capsize=10)
    # # axs[1,0].plot(BVal_holo[i],diff_m[i],marker="o",ls="solid",color=colors[i])
    axs[1,1].errorbar(BVal_holo[i]*1e3,amps_valley_vec[i],yerr=amps_valley_std_vec[i],marker="o",ls="solid",color=colors[i],capsize=10)
    # axs[1,1].errorbar(BVal_holo[i],periodicity[i],yerr=periodicity_std[i],marker="o",color=colors[i],capsize=10)
    # axs[2,1].errorbar(BVal_holo[i]*1e3,periodicity_valley[i],color=colors[i],marker="o")

axs[0,0].plot(BVal_holo*1e3,sigmas_pk_vec,"k")
axs[1,0].plot(BVal_holo*1e3,amps_pk_vec,"k")

axs[0,1].plot(BVal_holo*1e3,sigmas_valley_vec,"k")
axs[1,1].plot(BVal_holo*1e3,amps_valley_vec,"k")
for n in range(2):
    axs[n,0].tick_params(axis="both",labelsize=14)
    axs[n,1].tick_params(axis="both",labelsize=14)
    axs[0,n].set_title(valley_pks_titles[n],fontsize=20)
    # axs[n].text(-0.05,0.99,lbls[n],horizontalalignment="right",verticalalignment="top",transform=axs[n].transAxes,fontsize=30,fontname="Arial")

for n in range(2):
    axs[-1,n].set_xlabel("Applied |H| [mT]",fontsize=20)
axs[0,0].set_ylabel("$\sigma  \ [nm]$",fontsize=20)
axs[1,0].set_ylabel("A [rad]",fontsize=20)
# axs[2,0].set_ylabel("$\lambda \ [nm]$",fontsize=20)
if save_figs:
    fig.savefig(save_path+"\\"+"OA_AllAppliedH_FittedParameters_ValleyPeaks" + save_add + ".png",bbox_inches="tight")

## Model 

#%%
def SqrtFunc(x,A,x0,C):
    return A/np.sqrt(1 - (x-x0)) + C
plt.close("all")
Bval_holo_copy = BVal_holo.copy()
sigmas_pk_vec_copy = sigmas_pk_vec.copy()


fig,axs = plt.subplots(tight_layout=True)
axs.plot(BVal_holo_sorted+Bval_holo_copy[0],sigmas_pk_vec_sorted,"k-o")

axs.plot(BVal_holo_fit+Bval_holo_copy[0],SqrtFunc(BVal_holo_fit,*theo_fit),"r--")
# 1/(1+sqrt(x))
#%%
X = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
Y = [ 0,   1,   1,    0,   1,   2,   2,   0,   1]

Z = [x for _,x in sorted(zip(Y,X))]
print(Z)  # ["a", "d", "h", "b", "c", "e", "i", "f", "g"]