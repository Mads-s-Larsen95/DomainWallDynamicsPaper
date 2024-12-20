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
path_proj = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\Off_Axis_EH\LineProfiles_Projected_Potential\Location_1'
folders = glob.glob(path_proj+"\\**")

Bx_Mat = []
By_Mat = []
Bval_Mat = []
for i,folder in enumerate(folders):
    files = glob.glob(folder+"\\**.npy")
    files = natsort.natsorted(files)
    
    for file in files:
        if i==0:
            if "Scale" not in file:
                Bx_Mat.append(np.load(file))
                Bval_Mat.append(float(file.split("\\")[-1].split("Perc")[0].split("_")[-1])/100*2)
            else:
                scale = np.load(file)
        if i==1:
            if "Scale" not in file:
                By_Mat.append(np.load(file))
        
#%%
def CoshFunc(x,A,x0,sigma,C):
    return A*sigma*np.log( np.cosh( (x-x0) / sigma ) )  + C

e_const = 1.602e-19
hbar = 1.05e-34
# u0 = 1.256*1e-6
M0 = 1.35 #T
B0 = M0
t0 = 140e-9

A_guess = e_const*B0*t0/(hbar)

x = np.linspace(0,len(phi_holo[0]),len(phi_holo[0]))*scale
x = x*1e-9 #in m
    
plt.close("all")
lbls = ["(a)","(b)","(c)","(d)","(e)"]
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images'

if "Location_C" in path_holo:
    save_add = "_locC"
if "Location_1" in path_holo:
    save_add = "_loc1"
fig,axs = plt.subplots(tight_layout=True,figsize=(4*6,6),ncols=4)
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
    
    axs[1].plot(x*1e9,p-p[0],color=colors[i])
    # p = phi_holo[i]
    bck = np.polyfit(x,p,2)
    
    axs[1].plot(x*1e9,np.polyval(bck,x)-np.polyval(bck,x)[0],color=colors[i],ls="dashed")
    # if i==0:
    #     axsHowTo[0].plot(x,p-p[0],color=colors[i],label="$\Delta \phi$")
    #     axsHowTo[0].plot(x,np.polyval(bck,x)-np.polyval(bck,x)[0],color="r",ls="dashed",label="Background")
        
    p = p - (x**2*bck[0] + bck[1]*x + bck[-1])
    if float(BVal_holo[i]*1e3 == 20):
        p20 = p
        
    N = 5
    # x = 
    # p = np.convolve(p, np.ones(N)/N, mode='valid')
    # x = np.convolve(x, np.ones(N)/N, mode='valid')
    
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
    axs[2].plot(x*1e9,p+i*np.min(p),color=colors[i])
    # axsBckSub[1].
    peaks, peak_properties = find_peaks(p, prominence = 2)#,width=1)
    pi = 1/(p-np.max(p)-50e-9)
    valleys, _ = find_peaks(pi,prominence = 0.5)
    
    axs[2].plot(x[peaks]*1e9,p[peaks]+i*np.min(p),"x",color=colors[i],alpha=0.4)
    axs[2].plot(x[valleys]*1e9,p[valleys]+i*np.min(p),"o",color=colors[i],alpha=0.4)
    
    axsOnlyLines.plot(x*1e9,p,color=colors[i],label="Applied |H| = {:.0f} mT".format(BVal_holo[i]*1e3))
    axsOnlyLines.plot(x[peaks]*1e9,p[peaks],"x",color=colors[i],alpha=0.4)
    axsOnlyLines.plot(x[valleys]*1e9,p[valleys],"o",color=colors[i],alpha=0.4)
    diff_val = np.mean(p[peaks])-np.mean(p[valleys])
    diff_m.append(diff_val)
    diff_std.append(np.mean([np.std(p[peaks]),np.std(p[valleys])]))
    
    if i==0:
        axsHowTo[0].plot(x*1e9,p,color=colors[i])
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

        sigma_fit_param = 20*1e-9
        # amp_fit_param = 5*1e-9#np.max(pfit2)#+np.min(pfit2)
        
        p0 = [A_guess,xfit[np.argmax(pfit2)],sigma_fit_param,2]#,2]

        p1,pcov = opt.curve_fit(CoshFunc,xfit,pfit2,p0=p0)
        residuals = pfit2- CoshFunc(xfit, *p1)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((pfit2-np.mean(pfit2))**2)
        r_squared = 1 - (ss_res / ss_tot)
        if r_squared > 0.9:
            fit_amp,fit_mu, fit_stdev,C = p1
            
            t = hbar*diff_val/(e_const*fit_stdev*1e-9*B0)
            
            axs[2].plot(xfit*1e9,CoshFunc(xfit,*p1)+i*np.min(p),color=colors[i],ls="dotted")
            axsOnlyLines.plot(xfit*1e9,CoshFunc(xfit,*p1),color=colors[i],ls="dotted")
            if i==0:
                axsHowTo[1].plot(xfit*1e9,pfit2,color="k",alpha=0.4,lw=3)
                axsHowTo[1].plot(xfit*1e9,CoshFunc(xfit,*p1),color="b",ls="dotted")
                
                axsHowTo[0].plot(fit_mu*1e9,p[pk]+i*np.min(p),"x",color=colors[i],alpha=0.4)#,label="Peaks")
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

        sigma_fit_param = 20*1e-9
        # amp_fit_param = 5*1e-9#np.max(pfit2)#+np.min(pfit2)
        
        p0 = [A_guess,xfit[np.argmax(pfit2)],sigma_fit_param,np.max(pfit2)]#,2]

        p1,pcov = opt.curve_fit(CoshFunc,xfit,pfit2,p0=p0)
        residuals = pfit2- CoshFunc(xfit, *p1)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((pfit2-np.mean(pfit2))**2)
        r_squared = 1 - (ss_res / ss_tot)
        if r_squared > 0.9:
            fit_amp,fit_mu, fit_stdev,C = p1
            axs[2].plot(xfit*1e9,CoshFunc(xfit,*p1)+i*np.min(p),color=colors[i],ls="dashed")
            axsOnlyLines.plot(xfit*1e9,CoshFunc(xfit,*p1),color=colors[i],ls="dashed")
            if i==0:
                axsHowTo[1].plot(xfit*1e9,CoshFunc(xfit,*p1),color="r",ls="dashed")
                axsHowTo[0].plot(fit_mu*1e9,p[pk]+i*np.min(p),"o",color=colors[i],alpha=0.4)
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
    
    sigmas_i = (sigmas_pk+sigmas_val)/2
    sigmas_i_std = (sigmas_pk_std+sigmas_val_std)/2
    
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


axs[0].set_ylabel("Applied |H| [mT]",fontsize=20)
axs[0].set_xlabel("Measurement #",fontsize=20)
# Another way of removing MIP
dphi_holo_20mT = 0.5*(phi_holo[1] - phi_holo[11])
dphi_holo_20mT = dphi_holo_20mT-np.polyval(np.polyfit(x,dphi_holo_20mT,1),x)
dphi_holo_40mT = 0.5*(phi_holo[2] - phi_holo[12])
dphi_holo_0mT = 0.5*(phi_holo[0] - phi_holo[10])
dphi_holo_120mT = 0.5*(phi_holo[6] - phi_holo[16])
axs[3].plot(x*1e9,p20-p20[0],color="k",lw="2",label="Background Subtraction")
axs[3].plot(x*1e9,dphi_holo_20mT,color="r",lw="2",label="Difference of holograms")
# axs[3].plot(x,dphi_holo_40mT-dphi_holo_40mT[0],color="r",lw="2",label="40 mT")
# axs[3].plot(x,dphi_holo_120mT-dphi_holo_120mT[0],color="m",lw="2",label="120 mT")
axs[3].legend(loc="lower left",fontsize=14)
for n in range(4):
    axs[n].tick_params(axis="both",labelsize=14)
    axs[n].text(-0.15,0.99,lbls[n],horizontalalignment="right",verticalalignment="top",transform=axs[n].transAxes,fontsize=30,fontname="Arial")
    if n>0:
        axs[n].set_xlim([x[0]*1e9,x[-1]*1e9])
        axs[n].set_xlabel("Distance [nm]",fontsize=20)
        axs[n].set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
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
        
        axsHowTo[n].set_xlim([x[0]*1e9,x[-1]*1e9])
    else:
        axsHowTo[n].set_xlim([freqs_Halfed[0],freqs_Halfed[-1]])


axsHowTo[0].set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
axsHowTo[2].set_ylabel("Normalized Intensity",fontsize=20)
axsHowTo[2].set_xlabel("Frequency [1/nm]",fontsize=20)
# figHowTo.savefig(save_path+"\\"+"OA_HowTo" + save_add + ".png",bbox_inches="tight")

# ##############
# axsHstr.tick_params(axis="both",labelsize=14)
# axsHstr.set_ylabel("Applied |H| [mT]",fontsize=20)
# axsHstr.set_xlabel("Measurement #",fontsize=20)

# figHstr.savefig(save_path+"\\"+"AppliedH" + save_add + ".png",bbox_inches="tight")

# ####¤####################
# axsOnlyLines.set_xlabel("Distance [nm]",fontsize=20)
# axsOnlyLines.set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
# axsOnlyLines.tick_params(axis="both",labelsize=14)
# # axsOnlyLines.legend(loc="upper right",fontsize=16)
# axsOnlyLines.set_xlim([x[0],x[-1]])
# # figOnlyLines.savefig(save_path+"\\"+"AllProfiles_OnlyProfiles_WithBckSub" + save_add + ".png",bbox_inches="tight")
# ################################
t_m = np.asarray(t_m)
t_std = np.asarray(t_std)
amps_m = np.asarray(amps_m)
sigmas_m = np.asarray(sigmas_m)
fig,axs = plt.subplots(tight_layout=True,nrows=3,sharex=True,figsize=(10,8))
# axs.errorbar(BVal_holo,diff_m,yerr=diff_std,marker="o",ls="solid",color="k",capsize=10)
for i in range(len(BVal_holo)):
    
    axs[0].errorbar(BVal_holo[i]*1e3,sigmas_m[i]*1e9,yerr=sigmas_std[i]*1e9,marker="o",color=colors[i],ls="solid",capsize=10)
    # axs[1,0].plot(BVal_holo[i],diff_m[i],marker="o",ls="solid",color=colors[i])
    axs[1].errorbar(BVal_holo[i]*1e3,amps_m[i]*1e-9,yerr=amps_std[i]*1e-9,marker="o",ls="solid",color=colors[i],capsize=10)
    # axs[1,1].errorbar(BVal_holo[i],periodicity[i],yerr=periodicity_std[i],marker="o",color=colors[i],capsize=10)
    axs[2].errorbar(BVal_holo[i]*1e3,periodicity[i]*1e9,color=colors[i]*1e9,marker="o")
    
    
    # axs[3].errorbar(BVal_holo[i]*1e3,t_m[i]*1e9,yerr=t_std[i]*1e9,marker="o",ls="solid",color=colors[i],capsize=10)

axs[0].plot(BVal_holo*1e3,sigmas_m,"k")
axs[1].plot(BVal_holo*1e3,amps_m,"k")
axs[2].plot(BVal_holo*1e3,periodicity,"k")
# axs[3].plot(BVal_holo*1e3,t_m*1e9,"k")
for n in range(3):
    axs[n].tick_params(axis="both",labelsize=14)
    
    axs[n].text(0.01,0.99,lbls[n],horizontalalignment="left",verticalalignment="top",transform=axs[n].transAxes,fontsize=30,fontname="Arial")

axs[-1].set_xlabel("Applied |H| [mT]",fontsize=20)
axs[0].set_ylabel("$\sigma  \ [nm]$",fontsize=20)
axs[1].set_ylabel("A [$nm^{-1}$]",fontsize=20)
axs[2].set_ylabel("$\lambda \ [nm]$",fontsize=20)
# axs[3].set_ylabel("$Thickness \ [nm]$",fontsize=20)
# fig.savefig(save_path+"\\"+"OA_AllAppliedH_FittedParameters" + save_add + ".png",bbox_inches="tight")
# fig,axs = plt.subplots(tight_layout=True,nrows=4,sharex=True,figsize=(10,8))
# # axs.errorbar(BVal_holo,diff_m,yerr=diff_std,marker="o",ls="solid",color="k",capsize=10)
# for i in range(len(BVal_holo)):
    
#     axs[0].errorbar(BVal_holo[i]*1e3,sigmas_m[i],yerr=sigmas_std[i],marker="o",color=colors[i],ls="solid",capsize=10)
#     # axs[1,0].plot(BVal_holo[i],diff_m[i],marker="o",ls="solid",color=colors[i])
#     axs[1].errorbar(BVal_holo[i]*1e3,amps_m[i],yerr=amps_std[i],marker="o",ls="solid",color=colors[i],capsize=10)
#     # axs[1,1].errorbar(BVal_holo[i],periodicity[i],yerr=periodicity_std[i],marker="o",color=colors[i],capsize=10)
#     axs[2].errorbar(BVal_holo[i]*1e3,periodicity[i],color=colors[i],marker="o")
    
#     axs[3].errorbar(BVal_holo[i]*1e3,C_m[i],yerr=C_std[i],marker="o",ls="solid",color=colors[i],capsize=10)
    
# axs[0].plot(BVal_holo*1e3,sigmas_m,"k")
# axs[1].plot(BVal_holo*1e3,amps_m,"k")
# axs[2].plot(BVal_holo*1e3,periodicity,"k")

# axs[3].plot(BVal_holo*1e3,C_m,"k")
# for n in range(4):
#     axs[n].tick_params(axis="both",labelsize=14)
    
#     axs[n].text(-0.05,0.99,lbls[n],horizontalalignment="right",verticalalignment="top",transform=axs[n].transAxes,fontsize=30,fontname="Arial")

# axs[-1].set_xlabel("Applied |H| [mT]",fontsize=20)
# axs[0].set_ylabel("$\sigma  \ [nm]$",fontsize=20)
# axs[1].set_ylabel("A [rad]",fontsize=20)
# axs[2].set_ylabel("$\lambda \ [nm]$",fontsize=20)
# axs[3].set_ylabel("Offset [rad]",fontsize=20)
#%%

Bs = hbar*np.mean(amps_m)/(e_const*t0)
bstd = hbar*np.std(amps_m)/(e_const*t0)

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