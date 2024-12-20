# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:17:38 2024

@author: maslar
"""

import matplotlib.pyplot as plt
import numpy as np
import hyperspy.api as hs
import glob
from matplotlib.widgets import Cursor
from datetime import datetime
import cv2
import skimage
import scipy.ndimage as ndimage
from matplotlib.pyplot import cm
import natsort
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib as mpl
from scipy.signal import find_peaks
import matplotlib.animation as animation
def GetSB_And_Size(im):
    start = datetime.now()
    plt.close("all")
    fft = im.fft(True)
    fft_plot = fft.data
    rr,cc = np.shape(fft_plot)
    fig, axs = plt.subplots()
    axs.imshow(np.log(abs(fft_plot)), cmap="gray")
    cursor = Cursor(axs, useblit=True, color='k', linewidth=1)
    zoom_ok = False
    print('Zoom or pan to view, \npress spacebar when ready to click:')
    while not zoom_ok:
        zoom_ok = plt.waitforbuttonpress()
    print('\n Select center of sb')
    points = plt.ginput(1)
    points = np.array(points, dtype=int)
    sb_pos1 = points[0]
    #Create ROI with h=roi_w, w=roi_w (square) around the position chosen by user
    roi_w = 50
    fft_ROI = fft_plot[sb_pos1[1]-roi_w:sb_pos1[1]+roi_w,sb_pos1[0]-roi_w:sb_pos1[0]+roi_w]
    
    # Find maximum in ROI
    max_idx = np.unravel_index(np.log(abs(fft_ROI)).argmax(),fft_ROI.shape) #find max position [i,j]
    max_idx = np.asarray(max_idx[::-1]) #
    
    sb_pos1 = np.array([sb_pos1[1]-roi_w + max_idx[1],sb_pos1[0]-roi_w+max_idx[0]])
    

    x_len = int(abs(sb_pos1[1]-cc/2))
    y_len = int(abs(sb_pos1[0]-rr/2))
    L = np.sqrt(x_len**2+y_len**2)
    sb_size = L/3
    
    if sb_pos1[0] < rr/2 and sb_pos1[1] < cc/2: #Upper sb chosen to the left
        Txt = "Upper sb to the left"
        sb_position = np.array([y_len,x_len])
    if sb_pos1[0] < rr/2 and sb_pos1[1] > cc/2: #Upper sb chosen to the right
        Txt = "Upper sb to the left"
        sb_position = np.array([y_len,int(rr-x_len)])
    if sb_pos1[0] > rr/2 and sb_pos1[1] > cc/2: #Lower sb chosen to the right
        Txt = "Upper sb to the left"
        sb_position = np.array([int(rr-y_len),int(rr-x_len)])
    if sb_pos1[0] > rr/2 and sb_pos1[1] < cc/2: #Lower sb chosen to the left
        Txt = "Upper sb to the left"
        sb_position = np.array([int(rr-y_len),x_len])
    print("\n",Txt + ", points chosen : ", sb_pos1)
    axs.plot(sb_pos1[1], sb_pos1[0], 'r.')
    circle = plt.Circle((sb_pos1[1],sb_pos1[0]),
                        sb_size, color="r", lw=1.5, fill=False)
    axs.add_patch(circle)
    axs.axis("off")
    end = datetime.now()
    time_taken = end - start
    print('\n Time for function (%H:%M:%S.%ms): ', time_taken)
    return sb_position,sb_size

def reconstructed_func(im,imref,sb_position,sb_size,imsize):
    im_reconstruct_P = im.reconstruct_phase(imref,
        sb_position=sb_position, sb_size=sb_size, output_shape=(imsize[0]*1/4, imsize[1]*1/4))
    return im_reconstruct_P

def holography_process_unwrapped_phase_hyperspy(im_reconstruct_P):
    im_unwrap_P = im_reconstruct_P.unwrapped_phase()
    return im_unwrap_P.data
def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result,affine_mat
def crop_image(Message, image):
    coordinates = cv2.selectROI(Message, image)

    if coordinates != []:
        print("\n Succes! ROI has been selected.")
    cv2.destroyAllWindows()
    r1, c1 = int((coordinates[1])), int((coordinates[0]))  # Upper left corner
    # Lower right corner
    r2, c2 = (int(coordinates[1]+coordinates[3])
              ), (int(coordinates[0]+coordinates[2]))
    r1, c1 = int(r1), int(c1)
    r2, c2 = int(r2), int(c2)
    Matrix = [[r1, c1], [r2, c2]]
    cropped_img = image[(r1): (r2), (c1): (c2)]
    return cropped_img, Matrix

#%%

path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\Holography_Location1_ObjecteLensExc_2'

Files = glob.glob(path+"\\**.dm4")

Ref_Data = []
Obj_Data = []
for file in Files:
    if "_ref" in file:
        Ref_Data.append(file)
    if "_obj" in file:
        Obj_Data.append(file)
Obj_Data = natsort.natsorted(Obj_Data)
Ref_Data = natsort.natsorted(Ref_Data)
cal_factor = 0.0289415
Obj_Data_Stack = []
Ref_Data_Stack = []
BVal_all = []
for i in range(len(Obj_Data)):
    obj = hs.load(Obj_Data[i],signal_type="hologram")
    ref = hs.load(Ref_Data[i],signal_type="hologram")
    
    # obj,ref = obj*cal_factor,ref*cal_factor
    Obj_Data_Stack.append(obj)
    
    BVal = float(Obj_Data[i].split("Perc")[0].split("_")[-1])/100*2
    BVal_all.append(BVal)
    Ref_Data_Stack.append(ref)
#%%

Stack_To_Image = Obj_Data_Stack 

plt.close("all")
font_type = "Arial"
FontSize_Scale = 20

#############################################################################
font_scalebar = {'family': font_type,
                 'size': FontSize_Scale,
                 'weight': 'bold'}
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\Off_Axis_EH\Raw_Images\Location_1'
save_path_NP = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\Off_Axis_EH\Raw_Images\Location_1'

for i in range(len(Obj_Data)):
    obj = Obj_Data_Stack[i]
    scale = obj.axes_manager["x"].scale
    fig,axs = plt.subplots(tight_layout=True)
    BVal = float(Obj_Data[i].split("Perc")[0].split("_")[-1])/100*2
    if BVal > 0:
        if "rev" in Obj_Data[i]:
            text_add =  r'$\downarrow$'
        elif "rev" not in Obj_Data[i]:
            text_add = r'$\uparrow$'
    if BVal < 0:
        if "rev" in Obj_Data[i]:
            text_add =  r'$\uparrow$'
        elif "rev" not in Obj_Data[i]:
            text_add =  r'$\downarrow$'
    data = Stack_To_Image[i]
    
    im = obj.data
    axs.imshow(im,cmap="gray")
    axs.axis("off")
    
    scalebar = ScaleBar(scale,
                        obj.axes_manager[0].units, length_fraction=0.5,
                        location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                        )
    axs.add_artist(scalebar)
    # axs[i].set_title("Applied |B| = {:.1f}mT".format(BVals[i]*1E3),fontsize=30)
    axs.text(0.05, 0.95, "Applied |H|={:.1f} mT".format(BVal*1E3) + text_add, horizontalalignment='left',
     verticalalignment='top', transform=axs.transAxes,fontsize=20,color="w")
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    if i==0:
        fig2,axs2 = plt.subplots(tight_layout=True)
        axs2.imshow(im,cmap="gray")
        axs2.axis("off")
        scalebar = ScaleBar(scale,
                        obj.axes_manager[0].units, length_fraction=0.5,
                        location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                        )
        axs2.add_artist(scalebar)
        fig2.savefig(save_path+"\\"+save_Name+"_RawHologram_NoTxt.png",bbox_inches="tight")
    
    fig.savefig(save_path+"\\"+save_Name+"_RawHologram.png",bbox_inches="tight")
    
    np.save(save_path_NP + "\\" + save_Name+"_RawHologram.npy",im)
    np.save(save_path_NP+"\\"+"Scale.npy",scale)
    plt.close()
# axs[-1].

    
    
#%%
Reconstructed_Stack = []
sbpos_all = []
BVal_all = []
for i in range(len(Obj_Data_Stack)):
    obj = Obj_Data_Stack[i]
    ref = Ref_Data_Stack[i]
    sbpos = obj.estimate_sideband_position(ap_cb_radius=None,sb="upper")
    sbsiz = obj.estimate_sideband_size(sbpos)
    sbsiz = sbsiz*2/3
    rec_phase_img = reconstructed_func(obj,ref,sbpos,sbsiz,obj.data.shape)
    BVal = float(Obj_Data[i].split("Perc")[0].split("_")[-1])/100*2
    BVal_all.append(BVal)
    sbpos_all.append(sbpos)
    Reconstructed_Stack.append(rec_phase_img)

#%%
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\Holography_Location1_ObjecteLensExc_2_SavedFiles\FFTs'
plt.close("all")

im = Obj_Data_Stack[0]
fft = im.fft(True)
fft_plot = fft.data
rr,cc = np.shape(fft_plot)
sbpos = sbpos_all[i]
sbpos_data = sbpos.data
ylen = rr-sbpos_data[0]
xlen = sbpos_data[1]
L = np.sqrt(xlen**2+ylen**2)
sbpos_data_shift = np.array([rr/2 + ylen,cc/2-xlen])


fig, axs = plt.subplots(tight_layout=True)
axs.imshow(np.log(abs(fft_plot)), cmap="gray")
axs.plot(sbpos_data_shift[1],sbpos_data_shift[0],"r.")
circle = plt.Circle((sbpos_data_shift[1],sbpos_data_shift[0]),
                        sbsiz.data, color="r", lw=1.5, fill=False)
axs.add_patch(circle)
axs.set_ylim([rr/2+L+sbsiz.data+50,rr/2-L-sbsiz.data-50])
axs.set_xlim([cc/2-L-sbsiz.data-50,cc/2+L+sbsiz.data+50])
axs.axis("off")

scalebar = ScaleBar(1/scale,
                        "1/nm", length_fraction=0.5,dimension='si-length-reciprocal',
                        location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                        )
axs.add_artist(scalebar)
save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    # fig.savefig(save_path+"\\"+save_Name+"FFT.png",bbox_inches="tight")
    # plt.close()
#%%
save_path = r'C:\Users\maslar\OneDrive - Danmarks Tekniske Universitet\MowinShared\Project PhD Mads\MagneticFerrite_Finland\Results_Figs'
plt.close("all")
colors = cm.jet(np.linspace(0,1,len(Obj_Data)))
figPos,axsPos = plt.subplots(tight_layout=True)
axsPos.set_xlim([0,2048])
axsPos.set_ylim([2048,0])
axsins = axsPos.inset_axes([0.6,0.6,0.38,0.38])
for i in range(len(Obj_Data)):
    axsPos.imshow(np.log(abs(fft_plot)), cmap="gray")
    sbpos = sbpos_all[i]
    sbpos_data = sbpos.data
    ylen = rr-sbpos_data[0]
    xlen = sbpos_data[1]
    L = np.sqrt(xlen**2+ylen**2)
    sbpos_data_shift = np.array([rr/2 + ylen,cc/2-xlen])
    
    axsPos.plot(sbpos_data_shift[1],sbpos_data_shift[0],"o",color=colors[i])
    axsins.plot(sbpos_data_shift[1],sbpos_data_shift[0],"o",color=colors[i])
axsPos.indicate_inset_zoom(axsins, edgecolor="black")
axsPos.set_ylabel("Y [px]",fontsize=25)
axsPos.set_xlabel("X [px]",fontsize=25)
axsPos.tick_params(axis="both",labelsize=16)
figPos.savefig(save_path+"\\Does_SBPosMove.png",bbox_inches="tight")
#%%
## Plot Only Phase
Stack_To_Image = Reconstructed_Stack 
scale = Reconstructed_Stack[0].axes_manager["x"].scale
plt.close("all")
font_type = "Arial"
FontSize_Scale = 20

#############################################################################
font_scalebar = {'family': font_type,
                 'size': FontSize_Scale,
                 'weight': 'bold'}
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\Off_Axis_EH\Reconstructed_Phase\Location_1'
save_path_NP = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\Off_Axis_EH\Reconstructed_Phase\Location_1'
BVal_all = []
for i in range(len(Obj_Data)):
    fig,axs = plt.subplots(tight_layout=True)
    BVal = float(Obj_Data[i].split("Perc")[0].split("_")[-1])/100*2
    BVal_all.append(BVal)
    if BVal > 0:
        if "rev" in Obj_Data[i]:
            text_add =  r'$\downarrow$'
        elif "rev" not in Obj_Data[i]:
            text_add = r'$\uparrow$'
    if BVal < 0:
        if "rev" in Obj_Data[i]:
            text_add =  r'$\uparrow$'
        elif "rev" not in Obj_Data[i]:
            text_add =  r'$\downarrow$'
    data = Stack_To_Image[i]
    
    im = data.phase()
    axs.imshow(im,cmap="inferno")
    axs.axis("off")
    
    scalebar = ScaleBar(scale,
                        Reconstructed_Stack[i].axes_manager[0].units, length_fraction=0.5,
                        location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                        )
    axs.add_artist(scalebar)
    # axs[i].set_title("Applied |B| = {:.1f}mT".format(BVals[i]*1E3),fontsize=30)
    axs.text(0.05, 0.95, "Applied |H|={:.1f} mT".format(BVal*1E3) + text_add, horizontalalignment='left',
     verticalalignment='top', transform=axs.transAxes,fontsize=20,color="w")
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    
    
    # fig.savefig(save_path+"\\"+save_Name+"Reconstructed_Phase.png",bbox_inches="tight")
    fig.savefig(save_path+"\\"+save_Name+"_Reconstructed_Phase.png",bbox_inches="tight")
    np.save(save_path_NP + "\\" + save_Name+"_Reconstructed_Phase.npy",im)
    np.save(save_path_NP+"\\"+"Scale.npy",scale)
    plt.close()
# axs[-1].
#%%
## Plot Reconstructed Phase And Amplitude
Stack_To_Image = Reconstructed_Stack 
scale = Reconstructed_Stack[0].axes_manager["x"].scale
plt.close("all")
font_type = "Arial"
FontSize_Scale = 20

#############################################################################
font_scalebar = {'family': font_type,
                 'size': FontSize_Scale,
                 'weight': 'bold'}
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\Holography_Location1_ObjecteLensExc_2_SavedFiles\Reconstructed'
BVal_all = []
for i in range(len(Obj_Data)):
    imfigs = []
    fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(12,6))
    BVal = float(Obj_Data[i].split("Perc")[0].split("_")[-1])/100*2
    BVal_all.append(BVal)
    data = Stack_To_Image[i]
    
    RecPhasePlot = axs[0].imshow(data.phase(),cmap="inferno")
    axs[0].text(0.05, 0.95, "Applied |B|={:.1f} mT".format(BVal*1E3), horizontalalignment='left',
     verticalalignment='top', transform=axs[0].transAxes,fontsize=30,color="w")
    
    RecAmpPlot = axs[1].imshow(data.amplitude(),cmap="inferno",vmin=0,vmax=4)
    imfigs.append(RecPhasePlot)
    imfigs.append(RecAmpPlot)
    for n in range(2):
        axs[n].axis("off")
    
        scalebar = ScaleBar(scale,
                        Reconstructed_Stack[i].axes_manager[0].units, length_fraction=0.5,
                        location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                        )
        axs[n].add_artist(scalebar)
        divider = make_axes_locatable(axs[n])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cbar = plt.colorbar(imfigs[n],cax=cax,orientation="horizontal")
        if n==0:
            
            ticks = np.linspace(np.min(data.phase()), np.max(data.phase()), 5, endpoint=True)
            cbar.set_ticks(ticks)
            cbar.ax.set_xlabel("$[rad]$",fontsize=20)
            cbar.ax.set_xticklabels(
                        ['-$\pi$', r"-$\pi/2$", '0', '+$\pi/2$', '+$\pi$'])
        if n==1:
            cbar.ax.set_xlabel("Amplitude (a.u.)",fontsize=20)
        cbar.ax.tick_params(labelsize=16)
    # axs[i].set_title("Applied |B| = {:.1f}mT".format(BVals[i]*1E3),fontsize=30)
    
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    # fig.savefig(save_path+"\\"+save_Name+"Reconstructed_Phase_And_Amplitude.png",bbox_inches="tight")
    plt.close()
# %%


"""
Phase unwrapping section 
"""
Unwrapped_Phase_Stack = []
BVal_all = []
for i in range(len(Reconstructed_Stack)):
    Unwrapped_Phase_Stack += [holography_process_unwrapped_phase_hyperspy(Reconstructed_Stack[i])]
    BVal = float(Obj_Data[i].split("Perc")[0].split("_")[-1])/100*2
    BVal_all.append(BVal)

#%%


Stack_To_Image = Unwrapped_Phase_Stack 
scale = Reconstructed_Stack[0].axes_manager["x"].scale
new_scale = scale
plt.close("all")
font_type = "Arial"
FontSize_Scale = 20
#############################################################################
font_scalebar = {'family': font_type,
                 'size': FontSize_Scale,
                 'weight': 'bold'}

imfigs = []
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\Off_Axis_EH\Unwrapped_Phase\Location_1'
save_path_NP = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\Off_Axis_EH\Unwrapped_Phase\Location_1'

for i in range(len(Obj_Data)):
    fig,axs = plt.subplots(tight_layout=True)
    BVal = BVal_all[i]
    if BVal > 0:
        if "rev" in Obj_Data[i]:
            text_add =  r'$\downarrow$'
        elif "rev" not in Obj_Data[i]:
            text_add = r'$\uparrow$'
    if BVal < 0:
        if "rev" in Obj_Data[i]:
            text_add =  r'$\uparrow$'
        elif "rev" not in Obj_Data[i]:
            text_add =  r'$\downarrow$'
    data = Stack_To_Image[i]
    
    im = data
    imfig = axs.imshow(im,cmap="inferno",vmin=np.min(Stack_To_Image),vmax=np.max(Stack_To_Image))
    axs.axis("off")
    
    scalebar = ScaleBar(new_scale,
                        Reconstructed_Stack[i].axes_manager[0].units, length_fraction=0.5,
                        location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                        )
    axs.add_artist(scalebar)
    axs.text(0.05, 0.95, "Applied |H|={:.1f} mT".format(BVal*1E3) + text_add, horizontalalignment='left',
     verticalalignment='top', transform=axs.transAxes,fontsize=20,color="w")
    divider = make_axes_locatable(axs)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(imfig,cax=cax,orientation="vertical")
    cbar.ax.set_ylabel("$\Delta \phi$ [rad]",fontsize=30)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylim([np.min(Stack_To_Image),np.max(Stack_To_Image)])
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    
    fig.savefig(save_path+"\\"+save_Name+"_Unwrapped_Phase.png",bbox_inches="tight")
    np.save(save_path_NP + "\\" + save_Name+"_Unwrapped_Phase.npy",im)
    np.save(save_path_NP+"\\"+"Scale.npy",scale)
    plt.close()

#%%
from mpl_toolkits.axes_grid1 import ImageGrid
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images'
plt.close("all")
imfigs = []
lbls = ["(a)","(b)","(c)","(d)"]


fig,axs = plt.subplots(tight_layout=True,ncols=4,figsize=(16,6))
imfigs.append(axs[0].imshow(Obj_Data_Stack[0].data,cmap="gray"))
imfigs.append(axs[1].imshow(np.log(abs(fft_plot)),cmap="gray"))
circle = plt.Circle((sbpos_data_shift[1],sbpos_data_shift[0]),
                        sbsiz.data, color="r", lw=1.5, fill=False)
axs[1].add_patch(circle)
axs[1].plot(sbpos_data_shift[1],sbpos_data_shift[0],"r.")
axs[1].set_ylim([rr/2+L+sbsiz.data+50,rr/2-L-sbsiz.data-50])
axs[1].set_xlim([cc/2-L-sbsiz.data-50,cc/2+L+sbsiz.data+50])
scalebar = ScaleBar(1/new_scale,
                        "1/nm", length_fraction=0.5,dimension='si-length-reciprocal',
                        location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                        )
axs[1].add_artist(scalebar)
imfigs.append(axs[2].imshow(Reconstructed_Stack[0].phase(),cmap="inferno"))
imfigs.append(axs[3].imshow(Unwrapped_Phase_Stack[0],cmap="inferno"))

for n in range(4):
    axs[n].axis("off")
    if n!=1:
        print(n)
        if n==0:
            scale_i = Obj_Data_Stack[0].axes_manager["x"].scale
        else:
            scale_i = new_scale
    
        scalebar = ScaleBar(scale_i,
                        Reconstructed_Stack[i].axes_manager[0].units, length_fraction=0.5,
                        location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                        )
        axs[n].add_artist(scalebar)
        
    
    divider = make_axes_locatable(axs[n])
    cax = divider.new_vertical(size="5%", pad=0.05, pack_start=True)
    if n>1:
        fig.add_axes(cax)
        cbar = fig.colorbar(imfigs[n],cax=cax,orientation="horizontal")
        cbar.ax.set_xlabel("$\Delta \phi$ [rad]",fontsize=20)
        cbar.ax.tick_params(labelsize=14)
        if n==2:
            ticks = np.linspace(Reconstructed_Stack[0].phase().min(), Reconstructed_Stack[0].phase().max(), 5, endpoint=True)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(['-$\pi$', r"-$\pi/2$", '0', '+$\pi/2$', '+$\pi$'])
    axs[n].text(0.01,0.99,lbls[n],horizontalalignment="left",verticalalignment="top",transform=axs[n].transAxes,color="w",fontsize=30,fontname="Arial")
fig.savefig(save_path+"\\"+"OffAxis_ImageAnalysis.png",bbox_inches="tight")
   #%%
    
"""
Magnetization + Resize

"""
plt.close("all")
Stack_Copy = Unwrapped_Phase_Stack
scale = Reconstructed_Stack[0].axes_manager["x"].scale
new_scale = scale
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\Off_Axis_EH\Unwrapped_Phase_With_Magnetization\Location_1'
Magnetization_Stack = []
grad_stack = []
arrows_stack = []
for i,Img_cop in enumerate(Stack_Copy):
    BVal = BVal_all[i]
    if BVal >=  0:
        if "rev" in Obj_Data[i]:
            text_add =  r'$\downarrow$'
        elif "rev" not in Obj_Data[i]:
            text_add = r'$\uparrow$'
    if BVal < 0:
        if "rev" in Obj_Data[i]:
            text_add =  r'$\uparrow$'
        elif "rev" not in Obj_Data[i]:
            text_add =  r'$\downarrow$'
    Img_cop,rot_mat = rotate_image(Img_cop,-50)
    
    rot_mat2 = np.vstack((rot_mat,[0,0,1]))
    inv_rot = np.linalg.inv(rot_mat2)
    tr = mpl.transforms.Affine2D(inv_rot)
    if i==0:
        img_analyze = Img_cop/(np.max(Img_cop)-np.min(Img_cop))
        _,M = crop_image("Crop",img_analyze - np.min(img_analyze))
    
    roi = Img_cop[M[0][0]:M[1][0],M[0][1]:M[1][1]]
    Cropped_img = roi
    Magnetization_Stack.append(Stack_Copy[i])
    ax, ay = np.gradient(Cropped_img)

# norm = np.sqrt(ax**2 + ay**2)
# ax /= norm
# ay /= norm
    x = np.arange(M[0][1],M[1][1]) #/ new_scale
    y = np.arange(M[0][0],M[1][0]) #/ new_scale
    X,Y= np.meshgrid(x,y,indexing="xy")
    
    interval = 20
    
    len_rows = int(len(y)/interval) #rows
    len_cols = int(len(x)/interval) #cols
    
    # Running average:
    arrows = np.zeros((2, len_rows, len_cols))
    
    for j in range(len_rows):
        for k in range(len_cols):
            
            if j == len_rows - 1:
                stopindexy = -1
            else:
                stopindexy = (j + 1) * interval
                
            if k == len_cols - 1:
                stopindexx = -1
            else:
                stopindexx = (k + 1) * interval
            print(stopindexy)
            arrows[1, j, k] = 1/interval * np.sum(ax[j * interval : stopindexy, 
                                                     k * interval : stopindexx],
                                                 axis=None)
            arrows[0, j, k] = 1/interval * np.sum(ay[j * interval : stopindexy, 
                                                     k * interval : stopindexx],
                                                 axis=None)
    # arrows = np.asarray(arrows)
    arrows_x = np.asarray(arrows[1])
    arrows_y = np.asarray(arrows[0])
    # arrows_x,_ = rotate_image(arrows_x,50)
    # arrows_y,_ = rotate_image(arrows_y,50)
    
    # Draw arrows with equal length:
    norm = np.sqrt(arrows[0]**2 + arrows[1]**2)
    # arrows /= norm
    
    # Coordinates of arrows:
    phi_cols  = np.asarray(X[::interval, ::interval][:-1,:-1])
    phi_rows = np.asarray(Y[::interval, ::interval][:-1,:-1])
                  #+ 0.5 * interval)
    # phi_cols,_ = rotate_image(phi_cols,50)
    # phi_rows,_ = rotate_image(phi_rows,50)
    grad_stack.append([phi_cols,phi_rows])
    arrows_stack.append(arrows)
    
    fig,axs = plt.subplots(tight_layout=True)
    axs.imshow(Magnetization_Stack[i],cmap="inferno",origin="upper")
    
    ts = axs.transData
    axs.quiver(phi_cols,phi_rows,arrows_x,-arrows_y,pivot='mid',color="w",transform = tr + ts)
    axs.set_xlim([0,Magnetization_Stack[i].shape[1]])
    axs.set_ylim([Magnetization_Stack[i].shape[0],0])
    axs.axis("off")
    scalebar = ScaleBar(new_scale,
                        Reconstructed_Stack[i].axes_manager[0].units, length_fraction=0.5,
                        location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                        )
    axs.add_artist(scalebar)
    
    axs.text(0.05, 0.95, "Applied |H|={:.1f} mT".format(BVal*1E3) + text_add, horizontalalignment='left',
     verticalalignment='top', transform=axs.transAxes,fontsize=20,color="w")
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
   
    # if i==0:
    #     break
    fig.savefig(save_path+"\\"+save_Name+"Unwrapped_Phase_WithMagnetization.png",bbox_inches="tight")
    plt.close(fig)
    
    
#%%
#Line profile
plt.close("all")
scale = Reconstructed_Stack[0].axes_manager["x"].scale
new_scale = scale
FontSize_Scale = 20
font_type = "Arial"
#############################################################################
font_scalebar = {'family': font_type,
                 'size': FontSize_Scale,
                 'weight': 'bold'}

Stack_To_Analyze = Unwrapped_Phase_Stack
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\Off_Axis_EH\LineProfiles_Phase\Location_1'
save_path_NP = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\Off_Axis_EH\LineProfiles_Phase\Location_1'

colors = cm.rainbow(np.linspace(0,1,len(Obj_Data)))
phi_all = []
for i,im in enumerate(Stack_To_Analyze):
    BVal = BVal_all[i]
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    if BVal > 0:
        if "rev" in Obj_Data[i]:
            text_add =  r'$\downarrow$'
        elif "rev" not in Obj_Data[i]:
            text_add = r'$\uparrow$'
    if BVal < 0:
        if "rev" in Obj_Data[i]:
            text_add =  r'$\uparrow$'
        elif "rev" not in Obj_Data[i]:
            text_add =  r'$\downarrow$'
    im_rot,mat_rot = rotate_image(im,-220)
    if i==0:
        # im_rot,mat_rot = rotate_image(im,-220)
        rot_mat2 = np.vstack((mat_rot,[0,0,1]))
        inv_rot  = np.linalg.inv(rot_mat2)

        im_analyze = im_rot/(np.max(im_rot)-np.min(im_rot))
        im_analyze = im_analyze - np.min(im_analyze)
        _,mat = crop_image("Crop",im_analyze)
        c0,c1 = mat[0][1], mat[1][1]
        r0,r1 = mat[0][0], mat[1][0]
        fig,axs = plt.subplots(tight_layout=True,figsize=(8,6))
        imfig = axs.imshow(im,cmap="inferno",vmin=np.min(Stack_To_Analyze),vmax=np.max(Stack_To_Analyze))
        axs.axis("off")
        scalebar = ScaleBar(new_scale,
                        Reconstructed_Stack[i].axes_manager[0].units, length_fraction=0.5,
                        location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                        )
        axs.add_artist(scalebar)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(imfig,cax=cax,orientation="vertical")
        cbar.ax.set_ylabel("$\Delta \phi$ [rad]",fontsize=30)
        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_ylim([np.min(Stack_To_Analyze),np.max(Stack_To_Analyze)])
        
        tr = mpl.transforms.Affine2D(inv_rot) #Compute an matplotlib object, inverse matrix = inv_rot
        ts = axs.transData
        
        #Apply matrix with transform first by rotation, then translation w.r.t axs (ts)
        rec_rot = plt.Rectangle((c0, r0), c1-c0,r1-r0, fill=False, color="w", ls="dashed", transform=tr + ts)
        # axs.annotate("", xy=(c0, (r1-r0)/2+r0), xytext=(c1-c0, (r1-r0)/2+r0), arrowprops=dict(arrowstyle="->"),color="w")
        axs.arrow((c1-c0)/2+c0-50/2, (r1-r0)/2+r0, 50, 0, width = 2,color="w",transform=tr+ts)
        axs.add_patch(rec_rot)
        # axs.add_patch(arrow_rot)
        fig.savefig(save_path+"\\"+save_Name+"_Unwrapped_Phase_ROI.png")#,bbox_inches="tight")
      
        
    
    ROI = im_rot[r0:r1,c0:c1]
    
    phi = np.mean(ROI,axis=0)
    
    
    # np.save(save_path_NP + "\\" + save_Name+"_Unwrapped_Phase.npy",phi)
    # np.save(save_path_NP+"\\"+"Scale.npy",scale)
    
    phi_all.append(phi)
    
#%%
p = phi_all[-1]
x = np.linspace(0,len(p),len(p))*new_scale
save_path_holo = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\TIE_location1\From_Holo'
np.save(save_path_holo + "\\0BMagnetization_phase.npy",p)
np.save(save_path_holo + "\\0BMagnetization_x.npy",x)
#%%
plt.close("all")
import scipy.optimize as opt

def CoshFunc(x,A,x0,sigma,C):
    phi = A*np.log(np.cosh((x-x0)/sigma)) + C
    return phi

t = 90E-9 #nm
hbar = 1.05457182*1E-34
econst = 1.602E-19
fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(12,6))
fig2,axs2 = plt.subplots(tight_layout=True,ncols=3,nrows=2,figsize=(6*3,6*2), sharey='row')#,sharey=True)
figHowto,axsHowto = plt.subplots(tight_layout=True,ncols=3,sharey=False,figsize=(6*3,6))
figPhiBckSub,axsPhiBckSub = plt.subplots(tight_layout=True,figsize=(10,8))
figPhiBckSubSep,axsPhiBckSubSep = plt.subplots(tight_layout=True,figsize=(10,8))
figHyst,axsHyst = plt.subplots(tight_layout=True)
# fig11,axs11 = plt.subplots(tight_layout=True)
# figfft,axsfft = plt.subplots(tight_layout=True)
x = np.linspace(0,len(phi_all[0]),len(phi_all[0]))*new_scale
x = np.asarray(x)
phi_all = np.asarray(phi_all)
# x = x+0.01
popts = []
BValTit = []
periodicity = []
pfits = []
FTs = []
BStrmaxs = []

colors = cm.jet(np.linspace(0,1,len(Obj_Data)))
scan_dist = 30
scan_dist_forward = scan_dist + 0
scan_dist_backward = scan_dist_forward 
what_peak = 1
sigmas_m = []
sigmas_std = []
periodicity_fromGauss = []

i_search = False#len(phi_all)-2
for i in range(len(phi_all)):
    p = phi_all[i]
    p = p-p[0]
    N = 2
    # x = 
    p = np.convolve(p, np.ones(N)/N, mode='valid')
    x = np.linspace(0,len(p),len(p))*new_scale
    
    axs[0].plot(x,p,color=colors[i])
    
    if i==0:
        axsHowto[0].plot(x,p,color=colors[i],label="Raw $\Delta \phi$")
    BVal = float(Obj_Data[i].split("Perc")[0].split("_")[-1])/100*2
    if BVal >= 0:
        text_add1 = "+"
        if "rev" in Obj_Data[i]:
            text_add =  r'$\downarrow$'
        elif "rev" not in Obj_Data[i]:
            text_add = r'$\uparrow$'
    if BVal < 0:
        text_add1 = "-"
        if "rev" in Obj_Data[i]:
            text_add =  r'$\uparrow$'
        elif "rev" not in Obj_Data[i]:
            text_add =  r'$\downarrow$' 
    
    axsHyst.plot(i,BVal*1E3,"o",color=colors[i])
    axsHyst.text(i+0.1,BVal*1E3,text_add,color="k",fontsize=14,horizontalalignment = "left", verticalalignment = "center")
    
    #####################################################
    bck = np.polyfit(x,p,2)
    pfit = p - np.polyval(bck,x)
    
    if i==0:
        add_extra = abs(np.max(pfit))
    
    pfits.append(pfit)
    peaks, peak_properties = find_peaks(pfit, prominence = 2)#,width=1)
    
    # Renormalize to a proper PDF
    sigmas = []
    mus = []
    amps = []
    # if i==0:
    #     axs11.plot(x,pfit/np.max(pfit))
    peaks = peaks[peaks>scan_dist]
    for j,pk in enumerate(peaks):
        if BVal_all[i]==max(BVal_all):
            if j==0:
                continue
        xfit = x[pk-scan_dist_backward:pk+scan_dist_forward]
        N = len(xfit)
        pfit2 = pfit - np.max(pfit)
        pfit2 = pfit2[pk-scan_dist_backward:pk+scan_dist_forward]
        pfit2 = pfit2-np.max(pfit2)
        # pfit3 = pfit2
        sigma_fit_param = 2
        # if abs(BVal*1e3) > 50:
        #     sigma_fit_param = 2
            
        # elif abs(BVal*1e3) <  50:
        #     sigma_fit_param = 1
        amp_fit_param = np.max(pfit2)+np.min(pfit2)
        # p0 = [pfit[pk],x[pk],50] # Inital guess is a normal distribution
        p0 = [amp_fit_param,xfit[np.argmax(pfit2)],sigma_fit_param,2]
        # errfunc = lambda p, x, y: gauss(x, p) - y # Distance to the target function
        # p1, success = opt.curve_fit(gauss, xfit,pfit3,p0=p0)
        p1,pcov = opt.curve_fit(CoshFunc,xfit,pfit2,p0=p0)
        
        residuals = pfit2 - CoshFunc(xfit, *p1)
        
        ss_res = np.sum(residuals**2)
        
        ss_tot = np.sum((pfit2-np.mean(pfit2))**2)
        
        r_squared = 1 - (ss_res / ss_tot)
        
        if r_squared < 0.95:
            continue
        # print(p1[-2])
        # if p1[0]==p0[0]:
        #     print(i)
        
        # p2,success = opt.curve_fit(CoshFunc,xfit,pfit2,p0=p1)
        # p2 = p1
        fit_amp,fit_mu, fit_stdev, C = p1
        
        # FWHM = abs(2*np.sqrt(2*np.log(2))*fit_stdev)
        sigmas.append(fit_stdev)
        mus.append(fit_mu)
        
        if i_search:
            if i==i_search:
                axsPhiBckSub.plot(x,pfit-np.max(pfit),color=colors[i],label="{:.1f} mT".format(BVal*1E3) + text_add)
                axsPhiBckSub.plot(x[peaks],pfit[peaks]-np.max(pfit),color=colors[i],marker="x",ls="None")
        else:
            axsPhiBckSub.plot(x,pfit-np.max(pfit)-i*add_extra,color=colors[i],alpha=0.4,label="{:.1f} mT".format(BVal*1E3) + text_add)
            axsPhiBckSub.plot(x[peaks],pfit[peaks]-np.max(pfit)-i*add_extra,color=colors[i],marker="x",ls="None")
        if i==0:
            axsHowto[0].plot(x,np.polyval(bck,x),color=colors[i],ls="dashed",label="Background")
            axsHowto[1].plot(x,pfit-np.max(pfit),color=colors[i],label="$\Delta \phi$ w/ bck. subtract")
            axsHowto[1].plot(x[peaks],pfit[peaks]-np.max(pfit),"x",color="r",label="Peaks")
            
        if i_search:
            if i==i_search:
                axsPhiBckSub.plot(xfit,CoshFunc(xfit,*p1),color=colors[i],ls="dashed")
        else:
            axsPhiBckSub.plot(xfit[np.argmax(pfit2)],pfit2[np.argmax(pfit2)]-i*add_extra,color=colors[i],marker="x",ls="None")
            axsPhiBckSub.plot(xfit,pfit2-i*add_extra,color=colors[i],ls="solid")
            axsPhiBckSub.plot(xfit,CoshFunc(xfit,*p1)-i*add_extra,color=colors[i],ls="dashed")
            
        if i_search:
            if i==i_search:
                axsPhiBckSub.plot(xfit,CoshFunc(xfit,*p1),color=colors[i],ls="dashed")
        else:
            axsPhiBckSubSep.plot(xfit[np.argmax(pfit2)],pfit2[np.argmax(pfit2)]-i*add_extra,color=colors[i],marker="x",ls="None")
            axsPhiBckSubSep.plot(xfit,pfit2-i*add_extra,color=colors[i],ls="solid")
            axsPhiBckSubSep.plot(xfit,CoshFunc(xfit,*p1)-i*add_extra,color=colors[i],ls="dashed")
        if i==0:
            axsHowto[1].plot(xfit,CoshFunc(xfit,*p1),ls="dotted",color=colors[i],label="Gauss Fit")
        if abs(BVal)*1E3 == 0:
        # if "rev" not in Obj_Data[i]:
            axs2[0,0].plot(xfit,pfit2,color=colors[i])
            axs2[0,0].plot(xfit,CoshFunc(xfit,*p1),ls="dashed",color=colors[i])
        if abs(BVal)*1E3 == 20:
        # if "rev" not in Obj_Data[i]:
            axs2[0,1].plot(xfit,pfit2,color=colors[i])
            axs2[0,1].plot(xfit,CoshFunc(xfit,*p1),ls="dashed",color=colors[i])
        if abs(BVal)*1E3 == 120:
        # if "rev" not in Obj_Data[i]:
            axs2[0,2].plot(xfit,pfit2,color=colors[i])
            axs2[0,2].plot(xfit,CoshFunc(xfit,*p1),ls="dashed",color=colors[i])
    sigmas_m.append(np.mean(sigmas))
    sigmas_std.append(np.std(sigmas))
    
    # periodicity_fromGauss.append(mus[1]-mus[0])
   
    nfft = len(pfit)
    

    FT = np.fft.fftshift(np.fft.fft(pfit))#[:nfft//2]
    FTs.append(FT)
    x_interval = x[1]-x[0]
    # FT[0] = 0
    freqs = np.arange(1/x_interval/-2,1/x_interval/2,1/x_interval/nfft)
    
    FT_norm_abs = abs(FT)/np.max(abs(FT))
    axs[1].plot(freqs,FT_norm_abs,color=colors[i],label="{:.1f} mT".format(BVal*1E3) + text_add)
    FT_Halfed = FT_norm_abs[nfft//2:]
    freqs_Halfed = freqs[nfft//2:]
    FT_Max = freqs_Halfed[np.argmax(FT_Halfed)]
    # print(FT_Max)
    periodicity.append(1/FT_Max)
    if i==0:
        axsHowto[2].plot(freqs,FT_norm_abs,color=colors[i],label="FFT of $\Delta \phi$ w/ bck. subtr.")
        axsHowto[2].plot(freqs_Halfed[np.argmax(FT_Halfed)],FT_Halfed[np.argmax(FT_Halfed)],"rx",label="Periodicity Frequency")
    
    # periodicity.append(x[peaks[1]]-x[peaks[0]])
    if abs(BVal)*1E3 == 0:
            axs2[1,0].plot(freqs,FT_norm_abs,color=colors[i],label= text_add)
            BValTit.append(BVal*1E3)
    if abs(BVal)*1E3 == 20:
            axs2[1,1].plot(freqs,FT_norm_abs,color=colors[i],label=text_add1 + text_add)
            BValTit.append(BVal*1E3)
    if (BVal)*1E3 == 120:
            axs2[1,2].plot(freqs,FT_norm_abs,color=colors[i],label=text_add1 + text_add)
            BValTit.append(BVal*1E3)

print(BVal_all[i_search]*1e3)
axsPhiBckSub.set_xlim([0,x[-1]])
axsPhiBckSub.set_xlabel("x [nm]",fontsize=25)
axsPhiBckSub.set_ylabel("$\Delta \phi$ [rad]",fontsize=25)
axsPhiBckSub.tick_params(axis="both",labelsize=16)
axsPhiBckSub.legend(loc="upper left",bbox_to_anchor=(1.05,1),fontsize=14)

axsPhiBckSubSep.set_xlim([0,x[-1]])
axsPhiBckSubSep.set_xlabel("x [nm]",fontsize=25)
axsPhiBckSubSep.set_ylabel("$\Delta \phi$ [rad]",fontsize=25)
axsPhiBckSubSep.tick_params(axis="both",labelsize=16)
axsPhiBckSubSep.legend(loc="upper left",bbox_to_anchor=(1.05,1),fontsize=14)


axs[0].set_xlim([0,x[-1]])
axs[0].set_xlabel("x [nm]",fontsize=25)
axs[0].set_ylabel("$\Delta \phi $ [rad]",fontsize=25)

# axs[0].set_title("Phase"
axs[1].set_ylabel("Normalized Amplitude",fontsize=25)
axs[1].set_xlabel("Frequency [1/nm]",fontsize=25)
axs[1].legend(loc="upper left",fontsize=12,bbox_to_anchor=(1.05,1))
for n in range(2):
    axs[n].tick_params(axis="both",labelsize=16)

axs2[0,0].set_ylabel("$\Delta \phi $ [rad]",fontsize=25)
axs2[1,0].set_ylabel("Normalized Amplitude",fontsize=25)
for n in range(3):
    axs2[0,n].set_xlim([0,x[-1]])
    axs2[0,n].set_xlabel("x [nm]",fontsize=25)
    axs2[0,n].tick_params(axis="both",labelsize=16)
    axs2[1,n].tick_params(axis="both",labelsize=16)
    axs2[1,n].set_xlabel("Frequency [1/nm]",fontsize=25)
    handles, labels = axs2[1,n].get_legend_handles_labels()
    axs2[0,n].legend(handles=handles,labels=labels,loc="upper left",fontsize=14,ncol=2)
    axs2[0,n].set_title("Applied |B| = {:.0f} mT".format(BVal_all[n]),fontsize=20)

axsHowto[0].legend(loc="lower left",fontsize=16)
handles, labels = axsHowto[1].get_legend_handles_labels()
axsHowto[1].legend(handles=handles[:3], labels=labels[:3],loc="lower left",fontsize=16)#("x [nm]",fontsize=25)
axsHowto[2].legend(loc="center left",fontsize=16)
# axsHowto[1].legend(loc="upper left",fontsize=16)

axsHowto[0].set_title("Remove Background",fontsize=25)
axsHowto[1].set_title("Find Peaks + Cosh",fontsize=25)
axsHowto[2].set_title("FFT",fontsize=25)
for n in range(2):
    axsHowto[n].set_xlim([0,x[-1]])
    axsHowto[n].set_xlabel("x [nm]",fontsize=25)
    axsHowto[n].set_ylabel("$\Delta \phi $ [rad]",fontsize=25)
axsHowto[-1].set_xlim([freqs[0],freqs[-1]])
axsHowto[-1].set_xlabel("Frequency [1/nm]",fontsize=25)
axsHowto[-1].set_ylabel("Normalized Amplitude",fontsize=25)
for n in range(3):
    axsHowto[n].tick_params(axis="both",labelsize=16)

axsHyst.set_xlabel("Measurement #",fontsize=25)
axsHyst.set_ylabel("B [mT]",fontsize=25)
axsHyst.tick_params(axis="both",labelsize=16)
#######################################################################################
BVal_all=np.asarray(BVal_all)
periodicity_fromGauss = np.asarray(periodicity_fromGauss)
periodicity = np.asarray(periodicity)
periodicity2 = [np.mean(periodicity) for i in range(len(periodicity))]
periodicity2 = np.asarray(periodicity2)

figperiodicity,axsperiodicity = plt.subplots(tight_layout=True)
axsperiodicity.plot(BVal_all*1E3,periodicity2,color="k",ls="-",marker="")
for i in range(len(sigmas_m)):
    axsperiodicity.plot(BVal_all[i]*1E3,periodicity2[i],color=colors[i],marker="o")
axsperiodicity.set_ylabel("$\lambda $ [nm]",fontsize=25)
axsperiodicity.set_xlabel("Applied |B| [mT]",fontsize=25)
axsperiodicity.tick_params(axis="both",labelsize=16)

sigmas_m = np.asarray(sigmas_m)
sigmas_std = np.asarray(sigmas_std)
figsigmas,axssigmas = plt.subplots(tight_layout=True)
axssigmas.plot(BVal_all*1E3,sigmas_m,color="k",ls="-",marker="")
for i in range(len(sigmas_m)):
    axssigmas.errorbar(BVal_all[i]*1E3,sigmas_m[i],yerr=sigmas_std[i],color=colors[i],marker="o",capsize=10)
axssigmas.set_ylabel("$\sigma$ [nm]" ,fontsize=25)
axssigmas.set_xlabel("Applied |B| [mT]",fontsize=25)
axssigmas.tick_params(axis="both",labelsize=16)

figPeriodAndsigmas,axsPeriodAndsigmas = plt.subplots(tight_layout=True,nrows=2,sharex=True,figsize=(6,8))
axsPeriodAndsigmas[0].plot(BVal_all*1E3,periodicity2,color="k",ls="-",marker="")
for i in range(len(sigmas_m)):
    axsPeriodAndsigmas[0].plot(BVal_all[i]*1E3,periodicity2[i],color=colors[i],marker="o")

axsPeriodAndsigmas[1].plot(BVal_all*1E3,sigmas_m,color="k",ls="-",marker="")
for i in range(len(sigmas_m)):
    axsPeriodAndsigmas[1].errorbar(BVal_all[i]*1E3,sigmas_m[i],yerr=sigmas_std[i],color=colors[i],marker="o",capsize=10)
# axsPeriodAndsigmas[1].errorbar(BVal_all*1E3,sigmass_m*1e3,yerr=sigmass_std*1E3,color="k",ls="-",marker="o",capsize=10)
axsPeriodAndsigmas[1].set_ylabel("$\sigma$ [nm]" ,fontsize=25)
axsPeriodAndsigmas[0].set_ylabel("$\lambda $ [nm]",fontsize=25)
axsPeriodAndsigmas[1].set_xlabel("Applied |B| [mT]",fontsize=25)
for n in range(2):
    axsPeriodAndsigmas[n].tick_params(axis="both",labelsize=16)
    

            
            
   #%% 
save_path = r'C:\Users\maslar\OneDrive - Danmarks Tekniske Universitet\MowinShared\Project PhD Mads\MagneticFerrite_Finland\Results_Figs'

save_path_NP = r'C:\Users\maslar\OneDrive - Danmarks Tekniske Universitet\MowinShared\Project PhD Mads\MagneticFerrite_Finland\Numpy'

# pfits = np.asarray(pfits)
np.save(save_path_NP+"\\phi_all_" + Files[0].split("\\")[-2] + ".npy",phi_all)
np.save(save_path_NP+"\\phi_bckRemoved_" + Files[0].split("\\")[-2] + ".npy",pfits)
np.save(save_path_NP+"\\x_" + Files[0].split("\\")[-2] + ".npy",x)
np.save(save_path_NP+"\\BVals_" + Files[0].split("\\")[-2] + ".npy",BVal_all)
np.save(save_path_NP+"\\FFTs" + Files[0].split("\\")[-2] + ".npy",FTs)
np.save(save_path_NP+"\\Freqs" + Files[0].split("\\")[-2] + ".npy",freqs)
np.save(save_path_NP+"\\sigmas_mean" + Files[0].split("\\")[-2] + ".npy",sigmas_m)
np.save(save_path_NP+"\\sigmas_std" + Files[0].split("\\")[-2] + ".npy",sigmas_std)
fig.savefig(save_path+"\\RawPhi_And_FFT_WithLegend.png",bbox_inches="tight")
fig2.savefig(save_path+"\\RawPhi_3Examples_And_FFT_WithLegend.png",bbox_inches="tight")
figHowto.savefig(save_path+"\\ExampleHowTo_0mTExample.png",bbox_inches="tight")
figperiodicity.savefig(save_path+"\\figperiodicity.png",bbox_inches="tight")
figPeriodAndsigmas.savefig(save_path+"\\figPeriodAndSigma_FromCosh.png",bbox_inches="tight")
figsigmas.savefig(save_path+"\\figSigma_fromCosh.png",bbox_inches="tight")
figHyst.savefig(save_path+"\\WhatAppliedB.png",bbox_inches="tight")
figPhiBckSub.savefig(save_path+"\\SeperatedPhis_WithBckSub.png",bbox_inches="tight")
#%%
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\Holography_Location1_ObjecteLensExc_2_SavedFiles\Projected_Potential'
plt.close("all")
hbar = 6.626*1E-34/(2*np.pi)
econst = 1.602E-19
front_fact = hbar/econst
# front_fact = 1
Stack_To_Image = Unwrapped_Phase_Stack 
Bx_Stack = []
By_Stack = []
for i,img in enumerate(Stack_To_Image):
    imfigs = []
    img = ndimage.gaussian_filter(img,sigma=2)
    BVal = float(Obj_Data[i].split("Perc")[0].split("_")[-1])/100*2
    # if i==0:
    fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(10,7))
    grady = np.gradient(img,axis=0)*1/(new_scale*1e-9)
    gradx = np.gradient(img,axis=1)*1/(new_scale*1e-9)
    Bx = -front_fact*grady*1e6
    By = front_fact*gradx*1e6
    
    if i==0:
        stdX = np.std(Bx)
        stdY = np.std(By)
        vminTot,vmaxTot = np.min(Bx)+3*stdX,np.max(Bx)
    # if i==0:
    imfigs.append(axs[0].imshow(Bx,cmap="inferno",vmin=vminTot,vmax=vmaxTot))
    imfigs.append(axs[1].imshow(By,cmap="inferno",vmin=vminTot,vmax=vmaxTot))
       # axs[0].text("$\B_{\perp,x}$",fontsize)
    axs[0].set_title("$B_{\perp,x}$",fontsize=30)
    axs[1].set_title("$B_{\perp,y}$",fontsize=30)
    axs[0].text(0.01, 0.99, "Applied |B|={:.1f} mT".format(BVal*1E3) + text_add, horizontalalignment='left',
    verticalalignment='top', transform=axs[0].transAxes,fontsize=20,color="w")
    for n in range(2):
        axs[n].axis("off")
        scalebar = ScaleBar(new_scale,
                       Reconstructed_Stack[i].axes_manager[0].units, length_fraction=0.5,
                       location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                       )
        axs[n].add_artist(scalebar)
        divider = make_axes_locatable(axs[n])
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cbar = plt.colorbar(imfigs[n],cax=cax,orientation="horizontal")
        cbar.ax.set_xlabel("$[T*\mu m]$",fontsize=20)
        cbar.ax.tick_params(labelsize=12)
        ts = axs[n].transData
        rec_rot = plt.Rectangle((c0, r0), c1-c0,r1-r0, fill=False, color="w", ls="dashed", transform=tr + ts)
            
            # axs[n].add_patch(rec_rot)
                # cbar.ax.set_xlim([np.min(Stack_To_Image),np.max(Stack_To_Image)])
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    fig.savefig(save_path+"\\"+save_Name+"_Projected_Potential.png",bbox_inches="tight")

    plt.close()
    
    Bx_Stack.append(Bx)
    By_Stack.append(By)
    
#%%
#Line profile
plt.close("all")
scale = Reconstructed_Stack[0].axes_manager["x"].scale
new_scale = scale
FontSize_Scale = 20
font_type = "Arial"
#############################################################################
font_scalebar = {'family': font_type,
                 'size': FontSize_Scale,
                 'weight': 'bold'}

Stack_To_Analyze = Bx_Stack
fig_plot,axs_plot = plt.subplots(tight_layout=True)

colors = cm.rainbow(np.linspace(0,1,len(Obj_Data)))
Bx_all = []
By_all = []
for i,im in enumerate(Stack_To_Analyze):
    # if "Finale" in Obj_Data[i]:
    #     continue
    Bx_Rot,_ = rotate_image(im,-220)
    By_Rot,_ = rotate_image(By_Stack[i],-220)
    if i==0:
        axs_plot.imshow(Bx_Rot,cmap="inferno")
        rec = plt.Rectangle((c0,r0),c1-c0,r1-r0,fill=False,color="w",ls="dashed")
        axs_plot.add_patch(rec)
    Bx_vec = np.mean(Bx_Rot[r0:r1,c0:c1],axis=0)
    By_vec = np.mean(By_Rot[r0:r1,c0:c1],axis=0)
    
    Bx_all.append(Bx_vec)
    By_all.append(By_vec)

#%%
plt.close("all")
import scipy.optimize as opt
min_max_x = []
min_max_y = []
fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(14,8),sharey=True)
figHowto,axsHowto = plt.subplots(tight_layout=True,ncols=1,figsize=(8,6))
for i in range(len(Bx_all)):
    Bx = Bx_all[i]
    Bx = Bx-Bx[0]
    By = By_all[i]
    By = By-By[0]
    
    x = np.linspace(0,len(Bx),len(Bx))*new_scale
    
    
    bck = np.polyfit(x,Bx,1)
    Bxfit = Bx - np.polyval(bck,x)
    Bxfit = Bxfit-np.min(Bxfit)
    bck = np.polyfit(x,By,1)
    Byfit = By - np.polyval(bck,x)
    Byfit = Byfit-Byfit[0]
    
    BVal = float(Obj_Data[i].split("Perc")[0].split("_")[-1])/100*2
    min_max_x.append(np.max(Bxfit)-np.min(Bxfit))
    min_max_y.append(np.max(Byfit)-np.min(Byfit))
    # print(np.max(Bxfit)-np.min(Bxfit))
    
    if BVal >= 0:
        text_add1 = "+"
        if "rev" in Obj_Data[i]:
            text_add =  r'$\downarrow$'
        elif "rev" not in Obj_Data[i]:
            text_add = r'$\uparrow$'
    if BVal < 0:
        text_add1 = "-"
        if "rev" in Obj_Data[i]:
            text_add =  r'$\uparrow$'
        elif "rev" not in Obj_Data[i]:
            text_add =  r'$\downarrow$' 
    if i==0:
        add_extra = abs(np.max(Bx)-np.min(Bx))
    
    axs[0].plot(x,Bxfit-i*add_extra,color=colors[i],label="{:.0f} mT".format(BVal*1e3)+text_add)
    axs[1].plot(x,Byfit-i*add_extra,color=colors[i],label="{:.0f} mT".format(BVal*1e3)+text_add)
    
    if i == 0:
        print("k")
        
        axsHowto.plot(x,(Bxfit-np.min(Bxfit))*1e3,color=colors[i])
        
    
    
    peaksX, peak_propertiesX = find_peaks(Bxfit, prominence = 5E-2)
    peaksY, peak_propertiesY = find_peaks(Byfit, prominence = 5E-2)
    
    # axs[0].plot(x[peaksX],Bxfit[peaksX]-i*add_extra,"rx")
    # axs[1].plot(x[peaksY],Byfit[peaksY]-0*add_extra,"rx")


axs[0].set_ylabel("$|B_\perp| \ [T * \mu m]$",fontsize=25)
axs[1].legend(loc="upper left",bbox_to_anchor=(1.05,1),fontsize=14)
for n in range(2):
    axs[n].tick_params(axis="both",labelsize=14)

print(np.mean(min_max_x),np.std(min_max_x))
print(np.mean(min_max_y),np.std(min_max_y))
#%%
save_path = r'C:\Users\maslar\OneDrive - Danmarks Tekniske Universitet\MowinShared\Project PhD Mads\MagneticFerrite_Finland\Results_Figs'

# pfits = np.asarray(pfits)

fig.savefig(save_path+"\\RawProjB_XandY.png",bbox_inches="tight")
figHowto.savefig(save_path+"\\ExampleHowTo_BxProj_0mTExample.png",bbox_inches="tight")

#%%
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\Holography_Location1_ObjecteLensExc_2_SavedFiles\Projected_Potential'
plt.close("all")
hbar = 1.05457182*1E-34
econst = 1.602E-19

Stack_To_Image = Unwrapped_Phase_Stack 
B_Stack = []
BVal_all = []
for i,img in enumerate(Stack_To_Image):
    imfigs = []
    img = ndimage.gaussian_filter(img,sigma=4)
    BVal = float(Obj_Data[i].split("Perc")[0].split("_")[-1])/100*2
    
    BVal_all.append(BVal)
    gradx = np.gradient(img,1/new_scale,axis=0)
    grady = np.gradient(img,1/new_scale,axis=1)
    B = -hbar/econst*(gradx-grady)*1E9*1E6
    # By = hbar/econst*grady*1E9*1E6
    
    if i==4:
        fig,axs = plt.subplots(tight_layout=True,ncols=1,figsize=(10,7))
        stdX = np.std(B)
        # stdY = np.std(By)
        vminTot,vmaxTot = np.min(B)+3*stdX,np.max(B)
        
        imfigs.append(axs.imshow(B,cmap="inferno",vmin=vminTot,vmax=vmaxTot))
    # imfigs.append(axs[1].imshow(By,cmap="inferno",vmin=vminTot,vmax=vmaxTot))
       # axs[0].text("$\B_{\perp,x}$",fontsize)
        axs.set_title("$B_{\perp,x}$",fontsize=30)
    # axs[1].set_title("$B_{\perp,y}$",fontsize=30)
        axs.text(0.01, 0.99, "Applied |B|={:.1f} mT".format(BVal*1E3) + text_add, horizontalalignment='left',
                verticalalignment='top', transform=axs.transAxes,fontsize=20,color="w")
        
        axs.axis("off")
        scalebar = ScaleBar(new_scale,
                       Reconstructed_Stack[i].axes_manager[0].units, length_fraction=0.5,
                       location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                       )
        axs.add_artist(scalebar)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cbar = plt.colorbar(imfigs[0],cax=cax,orientation="horizontal")
        cbar.ax.set_xlabel("$[\mu T \cdot  nm]$",fontsize=20)
        cbar.ax.tick_params(labelsize=12)
        ts = axs.transData
        rec_rot = plt.Rectangle((c0, r0), c1-c0,r1-r0, fill=False, color="w", ls="dashed", transform=tr + ts)
        
        axs.add_patch(rec_rot)
            # cbar.ax.set_xlim([np.min(Stack_To_Image),np.max(Stack_To_Image)])
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    # fig.savefig(save_path+"\\"+save_Name+"_Projected_Potential_WithROI.png",bbox_inches="tight")

    # plt.close()
    B_Stack.append(B)
    # Bx_Stack.append(Bx)
    # By_Stack.append(By)
    
#%%

path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\13032024_SampleE'
files_EELS = glob.glob(path+"\\**.emi")
files_EELS = natsort.natsorted(files_EELS)

File_Interest = "LocationD"
files_new = []
for file in files_EELS:
    
    if "EELS" in file:
        if File_Interest in file and "After" not in file:
            files_new.append(file)
print(files_new)
for file in files_new:
    if "STEM" not in file:
        EELS_Map = hs.load(file,signal_type="EELS")
    if "STEM" in file:
        print(file)
        STEM_Image = hs.load(file)
        
#%%
plt.close("all")
STEM_Image_data = STEM_Image.data

save_path = r'C:\Users\maslar\OneDrive - Danmarks Tekniske Universitet\MowinShared\Project PhD Mads\MagneticFerrite_Finland\Results_Figs\LocationC_Data'
imfigs = []
fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(10,8))
imfigs.append(axs[0].imshow(STEM_Image_data-np.min(STEM_Image_data),cmap="gray"))
# imfigs.append(axs[1].imshow(Unwrapped_Phase_Stack_Cropped[0],cmap="inferno"))
for n in range(2):
    axs[n].axis("off")
scalebar = ScaleBar(STEM_Image.axes_manager["x"].scale,
                       STEM_Image.axes_manager["x"].units, length_fraction=0.5,
                       location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                       )
axs[0].add_artist(scalebar)
divider = make_axes_locatable(axs[0])
cax = divider.append_axes("bottom", size="5%", pad=0.05)
cbar = plt.colorbar(imfigs[0],cax=cax,orientation="horizontal")
cbar.ax.set_xlabel("Intensity",fontsize=20)
cbar.ax.tick_params(labelsize=12)
scalebar = ScaleBar(new_scale,
                       Reconstructed_Stack[0].axes_manager[0].units, length_fraction=0.5,
                       location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                       )
axs[1].add_artist(scalebar)
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("bottom", size="5%", pad=0.05)
cbar = plt.colorbar(imfigs[1],cax=cax,orientation="horizontal")
cbar.ax.set_xlabel("$\Delta \phi \ [rad]$",fontsize=20)
cbar.ax.tick_params(labelsize=12)

axs[0].set_title("HAADF-STEM",fontsize=30)
axs[1].set_title("Unwrapped Phase",fontsize=30)
# fig.savefig(save_path+"\\UnwrappedPhase_With_HAADFSTEM.png",bbox_inches="tight")
# _,M = crop_image("Crop",STEM_Image_data)

# STEM_Image_data_crop = STEM_Image_data[M[0][0]:M[1][0],M[0][1]:M[1][1]]
# fig,axs = plt.subplots(tight_layout=True,ncols=2)
# axs[0].imshow(STEM_Image_data_crop,cmap="gray")
# axs[1].imshow(Unwrapped_Phase_Stack[0],cmap="inferno")

#%%
plt.close("all")

EELS_Map.align_zero_loss_peak()

ZLP_pos = EELS_Map.estimate_zero_loss_peak_centre()

tlambda_map = EELS_Map.estimate_thickness(ZLP_pos)

tlambda_map.plot()

tlambda_map_data = tlambda_map.data

vac_bck = tlambda_map_data[0,0]#np.mean([,:1])
tlambda_map_data = tlambda_map_data - vac_bck


t_map = tlambda_map_data#*lambda_val
# t_map = t_map - np.min(t_map)

fig,axs = plt.subplots(tight_layout=True,ncols=2)
axs[0].imshow(STEM_Image_data-np.min(STEM_Image_data),cmap="gray")
imfig = axs[1].imshow(tlambda_map_data,cmap="inferno",vmin=0)
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(imfig,cax=cax,orientation="vertical")
cbar.ax.set_ylabel("$t/\lambda$",fontsize=20)
cbar.ax.tick_params(labelsize=12)
#%%
plt.close("all")
lambda_val = 125.1 #nm

t_map = tlambda_map_data#*lambda_val
# t_map = t_map - np.min(t_map)

fig,axs = plt.subplots(tight_layout=True)
imfig = axs.imshow(t_map,cmap="inferno",vmin=0)
divider = make_axes_locatable(axs)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(imfig,cax=cax,orientation="vertical")
cbar.ax.set_ylabel("$t [ nm]$",fontsize=20)
cbar.ax.tick_params(labelsize=12)
#%%
fig,axs = plt.subplots(tight_layout=True)
imfig = axs.imshow(tlambda_map_data,cmap="inferno",vmin=0,vmax=np.max(tlambda_map_data))
divider = make_axes_locatable(axs)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(imfig,cax=cax,orientation="vertical")
cbar.ax.set_ylabel("$t/\lambda$",fontsize=20)
cbar.ax.tick_params(labelsize=12)
scalebar = ScaleBar(EELS_Map.axes_manager["x"].scale,
                               "nm", length_fraction=0.5,
                               location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                               )
axs.add_artist(scalebar)
# fig.savefig(save_path+"\\tlambda_Map.png",bbox_inches="tight")
#%%
"""
GIFS
"""
#%%

#######################################
###### ANIMATION of Reconstructed Phase###########
#######################################
plt.close("all")
Reconstructed_Phase_Stack_data = []
for img in Reconstructed_Stack:
    Reconstructed_Phase_Stack_data.append(img.phase())
Reconstructed_Phase_Stack_data = np.asarray(Reconstructed_Phase_Stack_data)

fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(12,6))
vminTot = np.min(Reconstructed_Phase_Stack_data)#+3*np.std(Reconstructed_Phase_Stack_data)
vmaxTot = np.max(Reconstructed_Phase_Stack_data)#-3*np.std(Reconstructed_Phase_Stack_data)
imfigs = []
RecPlot = axs[0].imshow(Reconstructed_Phase_Stack_data[0],cmap="inferno",vmin=vminTot,vmax=vmaxTot)
imfigs.append(RecPlot)
Bvalplot, = axs[1].plot([],[],"k-o")


for n in range(1):
    axs[n].set_xlim([0,512])
    axs[n].set_ylim([512,0])
    axs[n].axis("off")
    divider = make_axes_locatable(axs[n])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(imfigs[n],cax=cax,orientation="horizontal")
    if n==0:
        
        ticks = np.linspace(Reconstructed_Phase_Stack_data[0].min(), Reconstructed_Phase_Stack_data[0].max(), 5, endpoint=True)
        cbar.set_ticks(ticks)
        cbar.ax.set_xlabel("$[rad]$",fontsize=20)
        cbar.ax.set_xticklabels(
                    ['-$\pi$', r"-$\pi/2$", '0', '+$\pi/2$', '+$\pi$'])
    if n==1:
        cbar.ax.set_xlabel("Amplitude (a.u.)",fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    scalebar = ScaleBar(scale,
                   Reconstructed_Stack[0].axes_manager[0].units, length_fraction=0.5,
                   location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                   )
    axs[n].add_artist(scalebar)
    
Bval = BVal_all[i]*1e3
Bvalplot.set_data([i,Bval])#,"k-o")
axs[1].set_xlim([0,len(BVal_all)])
axs[1].set_ylim([np.min(BVal_all)*1e3-50,np.max(BVal_all)*1e3+50])
axs[1].set_xlabel("Measurement #",fontsize=25)
axs[1].set_ylabel("Applied |B| [mT]",fontsize=25)
axs[1].set_xticks(np.linspace(1,len(BVal_all),len(BVal_all)))
axs[1].tick_params(axis="both",labelsize=14)
x,y = [],[]
def update(i):
    x.append(i+1)
    y.append(BVal_all[i]*1e3)
    RecPlot.set_data(Reconstructed_Phase_Stack_data[i])
    Bvalplot.set_data(x,y)
    if i==len(BVal_all)-1:
        x.clear()
        y.clear()
    return RecPlot,Bvalplot,

ani = animation.FuncAnimation(fig, update, repeat=True,
                                    frames=int(np.shape(Reconstructed_Phase_Stack_data)[0]))#

save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\Holography_Location1_ObjecteLensExc_2_SavedFiles\GIFs'
# ani.save(save_path+"\\Reconstructed_Phase.gif")
#%%

#######################################
###### ANIMATION of Reconstructed Phase + Amp ###########
#######################################
plt.close("all")
Reconstructed_Phase_Stack_data = []
Reconstructed_Amplitude_Stack_data = []
for img in Reconstructed_Stack:
    Reconstructed_Phase_Stack_data.append(img.phase())
    Reconstructed_Amplitude_Stack_data.append(img.amplitude())
Reconstructed_Phase_Stack_data = np.asarray(Reconstructed_Phase_Stack_data)
Reconstructed_Amplitude_Stack_data = np.asarray(Reconstructed_Amplitude_Stack_data)

fig,axs = plt.subplots(tight_layout=True,ncols=3,figsize=(18,6))
vminTot = np.min(Reconstructed_Phase_Stack_data)#+3*np.std(Reconstructed_Phase_Stack_data)
vmaxTot = np.max(Reconstructed_Phase_Stack_data)#-3*np.std(Reconstructed_Phase_Stack_data)
vminTot_Amp = 0#np.min(Reconstructed_Amplitude_Stack_data)+100*np.std(Reconstructed_Amplitude_Stack_data)
vmaxTot_Amp = 5#np.max(Reconstructed_Amplitude_Stack_data)-100*np.std(Reconstructed_Amplitude_Stack_data)
imfigs = []
RecPlot = axs[0].imshow(Reconstructed_Phase_Stack_data[0],cmap="inferno",vmin=vminTot,vmax=vmaxTot)
imfigs.append(RecPlot)
AmpPlot = axs[1].imshow(Reconstructed_Phase_Stack_data[0],cmap="inferno",vmin=vminTot_Amp,vmax=vmaxTot_Amp)
imfigs.append(AmpPlot)
Bvalplot, = axs[2].plot([],[],"k-o")


for n in range(2):
    axs[n].set_xlim([0,512])
    axs[n].set_ylim([512,0])
    axs[n].axis("off")
    divider = make_axes_locatable(axs[n])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(imfigs[n],cax=cax,orientation="horizontal")
    if n==0:
        
        ticks = np.linspace(Reconstructed_Phase_Stack_data[0].min(), Reconstructed_Phase_Stack_data[0].max(), 5, endpoint=True)
        cbar.set_ticks(ticks)
        cbar.ax.set_xlabel("$[rad]$",fontsize=20)
        cbar.ax.set_xticklabels(
                    ['-$\pi$', r"-$\pi/2$", '0', '+$\pi/2$', '+$\pi$'])
    if n==1:
        cbar.ax.set_xlabel("Amplitude (a.u.)",fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    scalebar = ScaleBar(scale,
                   Reconstructed_Stack[0].axes_manager[0].units, length_fraction=0.5,
                   location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                   )
    axs[n].add_artist(scalebar)
    
Bval = BVal_all[i]*1e3
Bvalplot.set_data([i,Bval])#,"k-o")
axs[2].set_xlim([0,len(BVal_all)])
axs[2].set_ylim([np.min(BVal_all)*1e3-50,np.max(BVal_all)*1e3+50])
axs[2].set_xlabel("Measurement #",fontsize=25)
axs[2].set_ylabel("Applied |B| [mT]",fontsize=25)
axs[2].set_xticks(np.linspace(1,len(BVal_all),len(BVal_all)))
axs[2].tick_params(axis="both",labelsize=14)
x,y = [],[]
def update(i):
    x.append(i+1)
    y.append(BVal_all[i]*1e3)
    RecPlot.set_data(Reconstructed_Phase_Stack_data[i])
    AmpPlot.set_data(Reconstructed_Amplitude_Stack_data[i])
    Bvalplot.set_data(x,y)
    if i==len(BVal_all)-1:
        x.clear()
        y.clear()
    return RecPlot,AmpPlot,Bvalplot,

ani = animation.FuncAnimation(fig, update, repeat=True,
                                    frames=int(np.shape(Reconstructed_Phase_Stack_data)[0]))#

save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\Holography_Location1_ObjecteLensExc_2_SavedFiles\GIFs'
ani.save(save_path+"\\Reconstructed_Phase_And_Amplitude.gif")
#%%
#######################################
###### ANIMATION of Unwrapped Phase ###########
#######################################
plt.close("all")
fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(12,6))
vminTot = np.min(Unwrapped_Phase_Stack)+3*np.std(Unwrapped_Phase_Stack)
vmaxTot = np.max(Unwrapped_Phase_Stack)-3*np.std(Unwrapped_Phase_Stack)

Unwr_Plot = axs[0].imshow(Unwrapped_Phase_Stack[0],cmap="inferno",vmin=vminTot,vmax=vmaxTot)

Bvalplot, = axs[1].plot([],[],"k-o")



axs[0].set_xlim([0,512])
axs[0].set_ylim([512,0])
axs[0].axis("off")
divider = make_axes_locatable(axs[0])
cax = divider.append_axes("bottom", size="5%", pad=0.05)
cbar = plt.colorbar(Unwr_Plot,cax=cax,orientation="horizontal")
cbar.ax.set_xlabel("$[\mu T \cdot  nm]$",fontsize=20)
cbar.ax.tick_params(labelsize=12)
scalebar = ScaleBar(new_scale,
                   Reconstructed_Stack[0].axes_manager[0].units, length_fraction=0.5,
                   location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                   )
axs[0].add_artist(scalebar)

axs[1].set_xlim([0,len(BVal_all)])
axs[1].set_ylim([np.min(BVal_all)*1e3-50,np.max(BVal_all)*1e3+50])
axs[1].set_xlabel("Measurement #",fontsize=25)
axs[1].set_ylabel("Applied |B| [mT]",fontsize=25)
axs[1].set_xticks(np.linspace(1,len(BVal_all),len(BVal_all)))
axs[1].tick_params(axis="both",labelsize=14)
x,y = [],[]
def update(i):
    x.append(i+1)
    y.append(BVal_all[i]*1e3)
    Unwr_Plot.set_data(Unwrapped_Phase_Stack[i])
    Bvalplot.set_data(x,y)
    if i==len(BVal_all)-1:
        x.clear()
        y.clear()
    return Bvalplot,

ani = animation.FuncAnimation(fig, update, repeat=True,
                                    frames=int(np.shape(Unwrapped_Phase_Stack)[0]))#

save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\Holography_Location1_ObjecteLensExc_2_SavedFiles\GIFs'
# ani.save(save_path+"\\Unwrapped_Phase.gif")



#%%
#######################################
###### ANIMATION of PROJ. B ###########
#######################################
plt.close("all")
fig,axs = plt.subplots(tight_layout=True,ncols=3,figsize=(18,6))

imfigs = []
Bxplot = axs[0].imshow(Bx_Stack[0],cmap="inferno",vmin=vminTot,vmax=vmaxTot)
imfigs.append(Bxplot)
Byplot = axs[1].imshow(Bx_Stack[0],cmap="inferno",vmin=vminTot,vmax=vmaxTot)
imfigs.append(Byplot)
Bvalplot, = axs[2].plot([],[],"k-o")




axs[0].text(0.01,0.99,"$B_{\perp,x}$",fontsize=40,color="w",horizontalalignment="left",verticalalignment="top",transform=axs[0].transAxes)

axs[1].text(0.01,0.99,"$B_{\perp,y}$",fontsize=40,color="w",horizontalalignment="left",verticalalignment="top",transform=axs[1].transAxes)
for n in range(2):
    axs[n].set_xlim([0,512])
    axs[n].set_ylim([512,0])
    axs[n].axis("off")
    divider = make_axes_locatable(axs[n])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(imfigs[n],cax=cax,orientation="horizontal")
    cbar.ax.set_xlabel("$[T * \mu m]$",fontsize=20)
    cbar.ax.tick_params(labelsize=12)
scalebar = ScaleBar(new_scale,
                   Reconstructed_Stack[0].axes_manager[0].units, length_fraction=0.5,
                   location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                   )
axs[0].add_artist(scalebar)
# Bvalplot.set_data([i,Bval])#,"k-o")
axs[2].set_xlim([0,len(BVal_all)])
axs[2].set_ylim([np.min(BVal_all)*1e3-50,np.max(BVal_all)*1e3+50])
axs[2].set_xlabel("Measurement #",fontsize=25)
axs[2].set_ylabel("Applied |B| [mT]",fontsize=25)
axs[2].set_xticks(np.linspace(1,len(BVal_all),len(BVal_all)))
axs[2].tick_params(axis="both",labelsize=14)
x,y = [],[]
def update(i):
    x.append(i+1)
    y.append(BVal_all[i]*1e3)
    Bxplot.set_data(Bx_Stack[i])
    Byplot.set_data(By_Stack[i])
    Bvalplot.set_data(x,y)
    if i==len(BVal_all)-1:
        x.clear()
        y.clear()
    return Bxplot,Byplot,Bvalplot,

ani = animation.FuncAnimation(fig, update, repeat=True,
                                    frames=int(np.shape(Bx_Stack)[0]))#

writer = animation.PillowWriter(fps=5,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\Holography_Location1_ObjecteLensExc_2_SavedFiles\GIFs'
ani.save(save_path+"\\Projected_B.gif")

#%%
#######################################
###### ANIMATION of MAGNETIZATION ###########
#######################################
plt.close("all")
fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(12,6))

imfigs = []
Quivers = []
# MagPlot = axs[0].imshow(Magnetization_Stack[0],cmap="inferno",origin="upper")
# ts = axs[0].transData
# QuiverPlot = axs[0].quiver(grad_stack[0][0],grad_stack[0][1],arrows_stack[0][1],-arrows_stack[0][0],pivot='mid',color="w",transform = tr+ts)
# axs[0].set_xlim([0,Magnetization_Stack[0].shape[1]])
# axs[0].set_ylim([Magnetization_Stack[0].shape[0],0])
imfigs.append(MagPlot)
Quivers.append(QuiverPlot)
Bvalplot, = axs[1].plot([],[],"k-o")

# axs[1].text(0.01,0.99,"$B_{\perp,y}$",fontsize=40,color="w",horizontalalignment="left",verticalalignment="top",transform=axs[1].transAxes)

axs[0].set_xlim([0,512])
axs[0].set_ylim([512,0])
axs[0].axis("off")
divider = make_axes_locatable(axs[0])
cax = divider.append_axes("bottom", size="5%", pad=0.05)
cbar = plt.colorbar(MagPlot,cax=cax,orientation="horizontal")
cbar.ax.set_xlabel("$\Delta \phi \ [rad]$",fontsize=20)
cbar.ax.tick_params(labelsize=12)
scalebar = ScaleBar(new_scale,
                   Reconstructed_Stack[0].axes_manager[0].units, length_fraction=0.5,
                   location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                   )
axs[0].add_artist(scalebar)

axs[1].set_xlim([0,len(BVal_all)])
axs[1].set_ylim([np.min(BVal_all)*1e3-50,np.max(BVal_all)*1e3+50])
axs[1].set_xlabel("Measurement #",fontsize=25)
axs[1].set_ylabel("Applied |B| [mT]",fontsize=25)
axs[1].set_xticks(np.linspace(1,len(BVal_all),len(BVal_all)))
axs[1].tick_params(axis="both",labelsize=14)

x,y = [],[]
def update(i):
    axs[0].clear()
    x.append(i+1)
    y.append(BVal_all[i]*1e3)
    # MagPlot.set_data(Magnetization_Stack[i
    
    axs[0].imshow(Magnetization_Stack[i],cmap="inferno")
    ts = axs[0].transData
    axs[0].quiver(grad_stack[i][0],grad_stack[i][1],arrows_stack[i][1],-arrows_stack[i][0],pivot='mid',color="w",transform = ts+tr)
    axs[0].set_xlim([0,Magnetization_Stack[0].shape[1]])
    axs[0].set_ylim([Magnetization_Stack[0].shape[0],0])
    Bvalplot.set_data(x,y)
    if i==len(BVal_all)-1:
        x.clear()
        y.clear()
    # QuiverPlot.clear()
    return Bvalplot,

ani = animation.FuncAnimation(fig, update, repeat=True,
                                    frames=int(np.shape(Magnetization_Stack)[0]))#

writer = animation.PillowWriter(fps=5,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\Holography_Location1_ObjecteLensExc_2_SavedFiles\GIFs'
# ani.save(save_path+"\\UnwrappedPhase_With_Magnetization_Rot.gif")