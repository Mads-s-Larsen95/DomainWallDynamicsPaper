# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:17:14 2024

@author: maslar
"""

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

path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE_2ndDay\LocationD_Holo'

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

Obj_Data_Stack = []
Ref_Data_Stack = []

for i in range(len(Obj_Data)):
    obj = hs.load(Obj_Data[i],signal_type="hologram")
    ref = hs.load(Ref_Data[i],signal_type="hologram")
    
    Obj_Data_Stack.append(obj)
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
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\Off_Axis_EH\Raw_Images'
save_path_NP = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\Off_Axis_EH\Raw_Images\Location_C'
for i in range(len(Obj_Data)):
    obj = Obj_Data_Stack[i]
    scale = obj.axes_manager["x"].scale
    fig,axs = plt.subplots(tight_layout=True)
    BVal = float(Obj_Data[i].split("_")[-1].split("Obj")[0])/100*2
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
    axs.text(0.05, 0.95, "Applied |B|={:.1f} mT".format(BVal*1E3), horizontalalignment='left',
     verticalalignment='top', transform=axs.transAxes,fontsize=30,color="w")
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    fig.savefig(save_path+"\\"+save_Name+"_RawHologram.png",bbox_inches="tight")
    np.save(save_path_NP + "\\" + save_Name+"_RawHologram.npy",im)
    plt.close()
# axs[-1].

    
    
#%%
Reconstructed_Stack = []
sbpos_all = []
for i in range(len(Obj_Data_Stack)):
    obj = Obj_Data_Stack[i]
    ref = Ref_Data_Stack[i]
    sbpos = obj.estimate_sideband_position(ap_cb_radius=None,sb="upper")
    sbsiz = obj.estimate_sideband_size(sbpos)
    sbsiz = sbsiz*2/3
    rec_phase_img = reconstructed_func(obj,ref,sbpos,sbsiz,obj.data.shape)
    
    sbpos_all.append(sbpos)
    Reconstructed_Stack.append(rec_phase_img)


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
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\Off_Axis_EH\Reconstructed_Phase\Location_C'
save_path_NP = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\Off_Axis_EH\Reconstructed_Phase\Location_C'
BVal_all = []
for i in range(len(Obj_Data)):
    fig,axs = plt.subplots(tight_layout=True)
    BVal = float(Obj_Data[i].split("_")[-1].split("Obj")[0])/100*2
    BVal_all.append(BVal)
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
    axs.text(0.05, 0.95, "Applied |H|={:.1f} mT".format(BVal*1E3), horizontalalignment='left',
     verticalalignment='top', transform=axs.transAxes,fontsize=20,color="w")
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    fig.savefig(save_path+"\\"+save_Name+"Reconstructed_Phase.png",bbox_inches="tight")
    np.save(save_path_NP + "\\" + save_Name+"Reconstructed_Phase.npy",im)
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
save_path = r'C:\Users\maslar\OneDrive - Danmarks Tekniske Universitet\MowinShared\Project PhD Mads\MagneticFerrite_Finland\Results_Figs\LocationC_Data\Reconstructed'
BVal_all = []
for i in range(len(Obj_Data)):
    imfigs = []
    fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(12,6))
    BVal = float(Obj_Data[i].split("_")[-1].split("Obj")[0])/100*2
    BVal_all.append(BVal)
    data = Stack_To_Image[i]
    
    RecPhasePlot = axs[0].imshow(data.phase(),cmap="inferno")
    axs[0].text(0.05, 0.95, "{:.1f} mT".format(BVal*1E3), horizontalalignment='left',
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
    fig.savefig(save_path+"\\"+save_Name+"Reconstructed_Phase_And_Amplitude.png",bbox_inches="tight")
    plt.close()
# %%


"""
Phase unwrapping section 
"""
Unwrapped_Phase_Stack = []
for i in range(len(Reconstructed_Stack)):
    Unwrapped_Phase_Stack += [holography_process_unwrapped_phase_hyperspy(Reconstructed_Stack[i])]
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
BVal_all = []

save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\Off_Axis_EH\Unwrapped_Phase\Location_C'
save_path_NP = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\Off_Axis_EH\Unwrapped_Phase\Location_C'
for i in range(len(Obj_Data)):
    fig,axs = plt.subplots(tight_layout=True)
    
    BVal = float(Obj_Data[i].split("_")[-1].split("Obj")[0])/100*2
    BVal_all.append(BVal)
    
    data = Stack_To_Image[i]
    
    im = data
    imfig = plt.imshow(im,cmap="inferno",vmin=np.min(Stack_To_Image),vmax=np.max(Stack_To_Image))
    axs.axis("off")
    
    scalebar = ScaleBar(new_scale,
                        Reconstructed_Stack[i].axes_manager[0].units, length_fraction=0.5,
                        location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                        )
    axs.add_artist(scalebar)
    # axs[i].set_title("Applied |B| = {:.1f}mT".format(BVals[i]*1E3),fontsize=30)
    axs.text(0.05, 0.95, "Applied |H|={:.1f} mT".format(BVal*1E3), horizontalalignment='left',
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
    # axs.quiver(ax,-ay)
    # fig.savefig(save_path+"\\"+save_Name+"Unwrapped_Phase.png",bbox_inches="tight")

    plt.close()


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
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\Off_Axis_EH\LineProfiles_Phase\Location_C'
save_path_NP = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\Off_Axis_EH\LineProfiles_Phase\Location_C'
angle_rots = [-40,-40,-40]
colors = cm.rainbow(np.linspace(0,1,len(Obj_Data)))
phi_all = []
for i,im in enumerate(Stack_To_Analyze):
    BVal = BVal_all[i]
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    
    im_rot,mat_rot = rotate_image(im,angle_rots[i])
    # if i==0:
        # im_rot,mat_rot = rotate_image(im,-220)
    rot_mat2 = np.vstack((mat_rot,[0,0,1]))
    inv_rot  = np.linalg.inv(rot_mat2)
    # if i==0:
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
    axs.text(0.05, 0.95, "Applied |H|={:.1f} mT".format(BVal*1E3), horizontalalignment='left',
     verticalalignment='top', transform=axs.transAxes,fontsize=20,color="w")
    # axs.add_patch(arrow_rot)
    fig.savefig(save_path+"\\"+save_Name+"_Unwrapped_Phase_ROI.png",bbox_inches="tight")
      
        
    
    ROI = im_rot[r0:r1,c0:c1]
    
    phi = np.mean(ROI,axis=0)
    
    
    np.save(save_path_NP + "\\" + save_Name+"_Unwrapped_Phase.npy",phi)
    np.save(save_path_NP+"\\"+"Scale.npy",scale)
    
    phi_all.append(phi)
#%%
save_path = r'C:\Users\maslar\OneDrive - Danmarks Tekniske Universitet\MowinShared\Project PhD Mads\MagneticFerrite_Finland\Results_Figs\LocationC_Data'
plt.close("all")
hbar = 1.05457182*1E-34
econst = 1.602E-19

Stack_To_Image = Unwrapped_Phase_Stack
Unwrapped_Phase_Stack_Cropped = []
for i,img in enumerate(Stack_To_Image):
    if i==0:
        img_analyze = img/(np.max(img)-np.min(img))
        
        _,M = crop_image("Crop",img_analyze - np.min(img_analyze))
    
    roi = img[M[0][0]:M[1][0],M[0][1]:M[1][1]]
    Unwrapped_Phase_Stack_Cropped.append(roi)
Stack_To_Image = Unwrapped_Phase_Stack_Cropped
Bx_Stack = []
By_Stack = []
for i,img in enumerate(Stack_To_Image):
    imfigs = []
    img = ndimage.gaussian_filter(img,sigma=4)
    BVal = float(Obj_Data[i].split("_")[-1].split("Obj")[0])/100*2
    
    fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(10,7))
    gradx = np.gradient(img,axis=0)*1/(new_scale*1E-9)
    grady = np.gradient(img,axis=1)*1/(new_scale*1E-9)
    Bx = -hbar/econst*gradx*1E6
    By = hbar/econst*grady*1E6
    
    if i==0:
        stdX = np.std(Bx)
        stdY = np.std(By)
        vminTot,vmaxTot = -1,1#np.min(Bx)+3*stdX,np.max(Bx)
        
    imfigs.append(axs[0].imshow(Bx,cmap="inferno",vmin=vminTot,vmax=vmaxTot))
    imfigs.append(axs[1].imshow(By,cmap="inferno",vmin=vminTot,vmax=vmaxTot))
       # axs[0].text("$\B_{\perp,x}$",fontsize)
    axs[0].set_title("$B_{\perp,x}$",fontsize=30)
    axs[1].set_title("$B_{\perp,y}$",fontsize=30)
    axs[0].text(0.01, 0.99, "Applied |B|={:.1f} mT".format(BVal*1E3), horizontalalignment='left',
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
        cbar.ax.set_xlabel("$[T * \mu m]$",fontsize=20)
        cbar.ax.tick_params(labelsize=12)
        ts = axs[n].transData
        # rec_rot = plt.Rectangle((c0, r0), c1-c0,r1-r0, fill=False, color="w", ls="dashed", transform=tr + ts)
        
        # axs[n].add_patch(rec_rot)
            # cbar.ax.set_xlim([np.min(Stack_To_Image),np.max(Stack_To_Image)])
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    fig.savefig(save_path+"\\"+save_Name+"_Projected_Potential.png",bbox_inches="tight")

    plt.close()
    
    Bx_Stack.append(Bx)
    By_Stack.append(By)


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
imfigs.append(axs[1].imshow(Unwrapped_Phase_Stack_Cropped[0],cmap="inferno"))
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
fig.savefig(save_path+"\\UnwrappedPhase_With_HAADFSTEM.png",bbox_inches="tight")
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

vac_bck = np.mean(tlambda_map_data[:5,:5])
tlambda_map_data = tlambda_map_data - vac_bck

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
plt.close("all")
lambda_val = 125.1 #nm

t_map = tlambda_map_data*lambda_val
# t_map = t_map - np.min(t_map)

fig,axs = plt.subplots(tight_layout=True)
imfig = axs.imshow(t_map,cmap="inferno",vmin=0)
divider = make_axes_locatable(axs)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(imfig,cax=cax,orientation="vertical")
cbar.ax.set_ylabel("$t [ nm]$",fontsize=20)
cbar.ax.tick_params(labelsize=12)

#%%
"""
Big plot

"""
plt.close("all")
imfigs = []
scales = [[STEM_Image.axes_manager["x"].scale,new_scale],[EELS_Map.axes_manager["x"].scale,EELS_Map.axes_manager["x"].scale]]
# units = [STEM_Image.axes_manager["x"].units,Reconstructed_Stack[i].axes_manager[0].units,EELS_Map.axes_manager["x"].units,EELS_Map.axes_manager["x"].units]
NamePlots = [["HAADF-STEM","Unwrapped Phase"],["$t/\lambda$ Map","$t(x,y)$"]]
fig,axs = plt.subplots(tight_layout=True,ncols=2,nrows=2,figsize=(10,8))

imfigs.append(axs[0,0].imshow(STEM_Image_data-np.min(STEM_Image_data),cmap="gray"))
imfigs.append(axs[0,1].imshow(Unwrapped_Phase_Stack_Cropped[0],cmap="inferno"))
imfigs.append(axs[1,0].imshow(tlambda_map_data,cmap="inferno",vmin=0,vmax=np.max(tlambda_map_data)))
imfigs.append(axs[1,1].imshow(t_map,cmap="inferno",vmin=0))

divider = make_axes_locatable(axs[0,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(imfigs[0],cax=cax,orientation="vertical")
cbar.ax.set_ylabel("$Intensity$",fontsize=20)
cbar.ax.tick_params(labelsize=12)
divider = make_axes_locatable(axs[0,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(imfigs[1],cax=cax,orientation="vertical")
cbar.ax.set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
cbar.ax.tick_params(labelsize=12)
divider = make_axes_locatable(axs[1,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(imfigs[2],cax=cax,orientation="vertical")
cbar.ax.set_ylabel("$t/\lambda$",fontsize=20)
cbar.ax.tick_params(labelsize=12)
divider = make_axes_locatable(axs[1,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(imfigs[3],cax=cax,orientation="vertical")
cbar.ax.set_ylabel("$t \ [nm]$",fontsize=20)
cbar.ax.tick_params(labelsize=12)

for i in range(2):
    for j in range(2):
        axs[i,j].axis("off")
        
        scalebar = ScaleBar(scales[i][j],
                               "nm", length_fraction=0.5,
                               location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                               )
        axs[i,j].add_artist(scalebar)
        axs[i,j].text(0.05,0.95,NamePlots[i][j],verticalalignment="top",horizontalalignment="left",transform=axs[i,j].transAxes,
                      color="w",fontsize=20)
fig.savefig(save_path+"\\BigPlot_WithHAADF_UnwrPhase_EELS.png",bbox_inches="tight")

#%%
"""
Resize

"""

Img_cop = Unwrapped_Phase_Stack[0]

rotimg,_ = rotate_image(Img_cop,40)

img_analyze = rotimg/(np.max(rotimg)-np.min(rotimg))
_,M = crop_image("Crop",img_analyze - np.min(img_analyze))
    
roi = rotimg[M[0][0]:M[1][0],M[0][1]:M[1][1]]
Cropped_img = roi

ratio_maps_x = np.shape(Cropped_img)[1]/np.shape(t_map)[1]
ratio_maps_y = np.shape(Cropped_img)[0]/np.shape(t_map)[0]
t_map_2 = cv2.resize(t_map, None, fx = ratio_maps_x, fy = ratio_maps_y)
new_scale_EELS = (EELS_Map.axes_manager["x"].scale/ratio_maps_x)
#%%
imfigs = []
scales = [EELS_Map.axes_manager["x"].scale,new_scale_EELS,new_scale]
fig,axs = plt.subplots(tight_layout=True,ncols=3,figsize=(8,6))
imfigs.append(axs[0].imshow(t_map,cmap="inferno"))
imfigs.append(axs[1].imshow(t_map_2,cmap="inferno"))
imfigs.append(axs[2].imshow(Cropped_img,cmap="inferno"))
for n in range(3):
    axs[n].axis("off")
    scalebar = ScaleBar(scales[n],
                               "nm", length_fraction=0.5,
                               location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                               )
    axs[n].add_artist(scalebar)
    
    divider = make_axes_locatable(axs[n])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(imfigs[n],cax=cax,orientation="horizontal")
    if n<=1:
        cbar.ax.set_xlabel("$t \ [nm]$",fontsize=20)
    if n==2:
        cbar.ax.set_xlabel("$\Delta \phi \ [rad]$",fontsize=20)
    cbar.ax.tick_params(labelsize=12)
# fig.savefig(save_path+"\\Resize_EELS_Map.png",bbox_inches="tight")
#%%
plt.close("all")


imfigs = []
scales = [Reconstructed_Stack[0].axes_manager["x"].scale,new_scale_EELS]
Names = ["$\Delta \phi \ [rad]$","$t \ [nm]$"]
fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(8,6))
imfigs.append(axs[0].imshow(Cropped_img,cmap="inferno"))
imfigs.append(axs[1].imshow(t_map_2,cmap="inferno"))
for n in range(2):
    axs[n].axis("off")
    scalebar = ScaleBar(scales[n],
                               "nm", length_fraction=0.5,
                               location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                               )
    axs[n].add_artist(scalebar)
    
    divider = make_axes_locatable(axs[n])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(imfigs[n],cax=cax,orientation="horizontal")
    cbar.ax.set_xlabel(Names[n],fontsize=20)
    cbar.ax.tick_params(labelsize=12)
# fig.savefig(save_path+"\\Resize_EELS_Map.png",bbox_inches="tight")
#%%
plt.close("all")
t = np.max(t_map)
Stack_To_Image = Unwrapped_Phase_Stack
hbar = 1.05457182*1E-34 #J*s
econst = 1.602E-19 #C
Bx_witht_Stack = []
By_witht_Stack = []
for i in range(len(Stack_To_Image)):
    
    img = Stack_To_Image[i]
    gradx = np.gradient(img,axis=0)*1/(new_scale*1e-9) #rad/nm
    grady = np.gradient(img,axis=1)*1/(new_scale*1e-9) #rad/nm
    Bx = -hbar/econst*gradx*1E9#*1e9 #T*nm 
    By = hbar/econst*grady*1e9 #T*nm
    
    Bx_rot,_ = rotate_image(Bx,40)
    By_rot,_ = rotate_image(By,40)
    
    Bx_crop = Bx_rot[M[0][0]:M[1][0],M[0][1]:M[1][1]]
    By_crop = By_rot[M[0][0]:M[1][0],M[0][1]:M[1][1]]
    Bx_witht = Bx_crop/t_map_2
    By_witht = By_crop/t_map_2
    
    imfigs = []
    BVal = float(Obj_Data[i].split("_")[-1].split("Obj")[0])/100*2
    if i==0:
        stdX = np.std(Bx_witht)
        w_roi = 200
        vminTot,vmaxTot = np.min(Bx_witht[540:600,700:800]),np.max(Bx_witht[540:600,700:800])
    fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(10,7))
    imfigs.append(axs[0].imshow(Bx,cmap="inferno"))#,vmin=vminTot,vma(x=vmaxTot))
    imfigs.append(axs[1].imshow(By,cmap="inferno"))#,vmin=vminTot,vmax=vmaxTot))
       # axs[0].text("$\B_{\perp,x}$",fontsize)
    axs[0].set_title("$B_{\perp,x}$",fontsize=30)
    axs[1].set_title("$B_{\perp,y}$",fontsize=30)
    axs[0].text(0.01, 0.99, "Applied |B|={:.1f} mT".format(BVal*1E3), horizontalalignment='left',
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
        cbar.ax.set_xlabel("$[T]$",fontsize=20)
        cbar.ax.tick_params(labelsize=12)
        ts = axs[n].transData
        # rec_rot = plt.Rectangle((c0, r0), c1-c0,r1-r0, fill=False, color="w", ls="dashed", transform=tr + ts)
        
        # axs[n].add_patch(rec_rot)
            # cbar.ax.set_xlim([np.min(Stack_To_Image),np.max(Stack_To_Image)])
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    # fig.savefig(save_path+"\\"+save_Name+"_Bfield_WithEELSProcessingDone.png",bbox_inches="tight")

    # plt.close()
    Bx_witht_Stack.append(Bx_witht)
    By_witht_Stack.append(By_witht)
#%%
save_path = r'C:\Users\maslar\OneDrive - Danmarks Tekniske Universitet\MowinShared\Project PhD Mads\MagneticFerrite_Finland\Results_Figs\LocationC_Data'

plt.close("all")
Names_Bs = ["$B_{\perp,x}$","$B_{\perp,x}$"]
fig,axs = plt.subplots(tight_layout=True,ncols=2,sharey=True)
for i in range(len(Bx_witht_Stack)):
    Bx = Bx_witht_Stack[i]
    By = By_witht_Stack[i]
    
    Bx,_ = rotate_image(Bx,-50)
    By,_  = rotate_image(By,-50)
    
    if i==0:
        fig2,axs2 = plt.subplots(tight_layout=True)
        axs2.imshow(Bx,cmap="inferno",vmin=vminTot,vmax=vmaxTot)
        
        points = fig2.ginput(n=4)
        points = np.asarray(points)
        
        r0,c0 = int(points[0][1]),int(points[0][0])
        h_roi,w_roi = int(points[3][1]-points[0][1]),int(points[3][0]-points[0][0])
    fig3,axs3 = plt.subplots(tight_layout=True,ncols=2,nrows=2,sharey='row',figsize=(10,8))
    axs3[0,0].imshow(Bx,cmap="inferno",vmin=vminTot,vmax=vmaxTot)
    axs3[0,1].imshow(By,cmap="inferno",vmin=vminTot,vmax=vmaxTot)
    for n in range(2):
        rec = plt.Rectangle((c0,r0),w_roi,h_roi,fill=False,color="w",lw=1.5)
        axs3[0,n].add_patch(rec)
        axs3[0,n].axis("off")
        axs3[0,n].text(0.05,0.95,Names_Bs[n],horizontalalignment="left",verticalalignment="top",transform=axs3[0,n].transAxes,
                       color="w",fontsize=20)
    roi_x = Bx[r0:r0+h_roi,c0:c0+w_roi]
    roi_y = By[r0:r0+h_roi,c0:c0+w_roi]
    
    Bx_m = np.mean(roi_x,axis=0)
    Bx_m = Bx_m - Bx_m[0]
    By_m = np.mean(roi_y,axis=0)
    By_m = By_m - By_m[0]
    if i==0:
        x = np.linspace(0,len(Bx_m),len(Bx_m))*new_scale
    
    axs[0].plot(x,Bx_m)
    axs[1].plot(x,By_m)
    
    axs3[1,0].plot(x,Bx_m,"k")
    axs3[1,1].plot(x,By_m,"k")
    for n in range(2):
        axs3[1,n].set_xlabel("Distance [nm]",fontsize=20)
    axs3[1,0].set_ylabel("$B_\perp$",fontsize=20)
    
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    fig3.savefig(save_path+"\\"+save_Name+"_Potential_WithPlots.png",bbox_inches="tight")
for n in range(2):
    axs[n].tick_params(axis="both",labelsize=14)
    axs[n].set_xlabel("Distance [nm]",fontsize=20)
axs[0].set_ylabel("$B_\perp \ [nT]$",fontsize=20)
#%%
"""
Resize

"""
plt.close("all")
angle_rots = [40,40,20]
Stack_To_Analyze = Unwrapped_Phase_Stack
Cropped_Imgs = []
for i,img in enumerate(Stack_To_Analyze):
    rotimg,_ = rotate_image(img,angle_rots[i])
    fig,axs = plt.subplots(tight_layout=True,ncols=2)
    axs[0].imshow(t_map,cmap="inferno")
    axs[1].imshow(rotimg,cmap="inferno")
    
    if i==0:
        img_analyze = rotimg/(np.max(rotimg)-np.min(rotimg))
        _,M = crop_image("Crop",img_analyze - np.min(img_analyze))
    
    roi = rotimg[M[0][0]:M[1][0],M[0][1]:M[1][1]]
    Cropped_img = roi
    
    if i==0:
        ratio_maps_x = np.shape(Cropped_img)[1]/np.shape(t_map)[1]
        ratio_maps_y = np.shape(Cropped_img)[0]/np.shape(t_map)[0]
        t_map_2 = cv2.resize(t_map, None, fx = ratio_maps_x, fy = ratio_maps_y)
        new_scale_EELS = (EELS_Map.axes_manager["x"].scale/ratio_maps_x)
    fig,axs = plt.subplots(tight_layout=True,ncols=2)
    axs[0].imshow(t_map_2,cmap="inferno")
    axs[1].imshow(Cropped_img,cmap="inferno")
    
#%%
plt.close("all")
t = np.max(t_map)
Stack_To_Image = Unwrapped_Phase_Stack
Bx_witht_Stack = []
By_witht_Stack = []
angle_rots = [40,40,20]
for i in range(len(Stack_To_Image)):
    
    img = Stack_To_Image[i]
    gradx = np.gradient(img,axis=0)*1/(new_scale*1e-9)
    grady = np.gradient(img,axis=1)*1/(new_scale*1e-9)
    Bx = -hbar/econst*gradx*1E9 #T*nm
    By = hbar/econst*grady*1E9 #T*nm
    
    Bx_rot,_ = rotate_image(Bx,angle_rots[i])
    By_rot,_ = rotate_image(By,angle_rots[i])
    
    Bx_crop = Bx_rot[M[0][0]:M[1][0],M[0][1]:M[1][1]]
    By_crop = By_rot[M[0][0]:M[1][0],M[0][1]:M[1][1]]
    Bx_witht = Bx_crop/t_map_2
    By_witht = By_crop/t_map_2
    
    imfigs = []
    BVal = float(Obj_Data[i].split("_")[-1].split("Obj")[0])/100*2
    if i==0:
        stdX = np.std(Bx_witht)
        w_roi = 200
        vminTot,vmaxTot = np.min(Bx_witht[540:600,700:800]),np.max(Bx_witht[540:600,700:800])
    fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(10,7))
    imfigs.append(axs[0].imshow(Bx_witht,cmap="inferno",vmin=vminTot,vmax=vmaxTot))
    imfigs.append(axs[1].imshow(By_witht,cmap="inferno",vmin=vminTot,vmax=vmaxTot))
       # axs[0].text("$\B_{\perp,x}$",fontsize)
    axs[0].set_title("$B_{\perp,x}$",fontsize=30)
    axs[1].set_title("$B_{\perp,y}$",fontsize=30)
    axs[0].text(0.01, 0.99, "Applied |B|={:.1f} mT".format(BVal*1E3), horizontalalignment='left',
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
        cbar.ax.set_xlabel("$[T]$",fontsize=20)
        cbar.ax.tick_params(labelsize=12)
        ts = axs[n].transData
        # rec_rot = plt.Rectangle((c0, r0), c1-c0,r1-r0, fill=False, color="w", ls="dashed", transform=tr + ts)
        
        # axs[n].add_patch(rec_rot)
            # cbar.ax.set_xlim([np.min(Stack_To_Image),np.max(Stack_To_Image)])
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    fig.savefig(save_path+"\\"+save_Name+"_Bfield_WithEELSProcessingDone.png",bbox_inches="tight")

    # plt.close()
    Bx_witht_Stack.append(Bx_witht)
    By_witht_Stack.append(By_witht)
#%%
save_path = r'C:\Users\maslar\OneDrive - Danmarks Tekniske Universitet\MowinShared\Project PhD Mads\MagneticFerrite_Finland\Results_Figs\LocationC_Data'

plt.close("all")
Names_Bs = ["$B_{\perp,x}$","$B_{\perp,x}$"]
fig,axs = plt.subplots(tight_layout=True,ncols=2,sharey=True,figsize=(10,8))

diffBx_vec = []
diffBy_vec = []
BVal_vec = []
for i in range(len(Bx_witht_Stack)):
    BVal = float(Obj_Data[i].split("_")[-1].split("Obj")[0])/100*2
    BVal_vec.append(BVal)
    Bx = Bx_witht_Stack[i]
    By = By_witht_Stack[i]
    
    Bx,_ = rotate_image(Bx,-50)
    By,_  = rotate_image(By,-50)
    
    fig2,axs2 = plt.subplots(tight_layout=True)
    axs2.imshow(Bx,cmap="inferno",vmin=vminTot,vmax=vmaxTot)
    
    points = fig2.ginput(n=3)
    points = np.asarray(points)
    
    r0,c0 = int(points[0][1]),int(points[0][0])
    h_roi,w_roi = int(points[2][1]-points[0][1]),int(points[2][0]-points[0][0])
    plt.close(fig2)
    fig3,axs3 = plt.subplots(tight_layout=True,ncols=2,nrows=2,sharey='row',figsize=(10,8))
    axs3[0,0].imshow(Bx,cmap="inferno",vmin=vminTot,vmax=vmaxTot)
    axs3[0,1].imshow(By,cmap="inferno",vmin=vminTot,vmax=vmaxTot)
    for n in range(2):
        rec = plt.Rectangle((c0,r0),w_roi,h_roi,fill=False,color="w",lw=1.5)
        axs3[0,n].add_patch(rec)
        axs3[0,n].axis("off")
        axs3[0,n].text(0.05,0.95,Names_Bs[n],horizontalalignment="left",verticalalignment="top",transform=axs3[0,n].transAxes,
                       color="w",fontsize=20)
    roi_x = Bx[r0:r0+h_roi,c0:c0+w_roi]
    roi_y = By[r0:r0+h_roi,c0:c0+w_roi]
    
    Bx_m = np.mean(roi_x,axis=0)
    # Bx_m = Bx_m - Bx_m[0]
    By_m = np.mean(roi_y,axis=0)
    # By_m = By_m - By_m[0]
    
    x = np.linspace(0,len(Bx_m),len(Bx_m))*new_scale
    if i==2:
        a=0.2
    else:
        a = 1
    lbl = "{:.1f} mT".format(BVal*1e3)
    axs[0].plot(x,Bx_m,alpha=a)
    axs[1].plot(x,By_m,alpha=a,label=lbl)
    points2 = fig.ginput(n=4)
    points2 = np.asarray(points2)
    diffBx = points2[0][1]-points2[1][1]
    diffBy = points2[2][1]-points2[3][1]
    axs[0].plot(points2[0][0],points2[0][1],"kx")
    axs[0].plot(points2[1][0],points2[1][1],"kx")
    
    axs[1].plot(points2[2][0],points2[2][1],"kx")
    axs[1].plot(points2[3][0],points2[3][1],"kx")
    print(diffBx,diffBy)
    
    diffBx_vec.append(diffBx)
    diffBy_vec.append(diffBy)
    
    axs3[1,0].plot(x,Bx_m,"k")
    axs3[1,1].plot(x,By_m,"k")
    for n in range(2):
        axs3[1,n].set_xlabel("Distance [nm]",fontsize=20)
    axs3[1,0].set_ylabel("$B_\perp \ [T]$",fontsize=20)
    
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    fig3.savefig(save_path+"\\"+save_Name+"_Potential_WithPlots.png",bbox_inches="tight")
for n in range(2):
    axs[n].tick_params(axis="both",labelsize=14)
    axs[n].set_xlabel("Distance [nm]",fontsize=20)
axs[0].set_ylabel("$B_\perp \ [T]$",fontsize=20)
axs[1].legend(loc="upper left",bbox_to_anchor=(1.05,1),fontsize=14)
#%%
figBstr,axsStr = plt.subplots(tight_layout=True,figsize=(8,6))
BVal_vec = np.asarray(BVal_vec)
axsStr.plot(BVal_vec*1e3,diffBx_vec,"k-o",label="$B_{\perp,x}$")
axsStr.plot(BVal_vec*1e3,diffBy_vec,"k-s",label="$B_{\perp,y}$")
axsStr.set_xlabel("Applited B [mT]",fontsize=20)
axsStr.set_ylabel("$\Delta B_\perp$ [T]",fontsize=20)
axsStr.legend(loc="upper right",fontsize=16)
axsStr.tick_params(axis="both",labelsize=16)
axsStr.set_xticks(BVal_vec*1e3)
fig.savefig(save_path+"\\"+"_Potential_WithPlots_All3.png",bbox_inches="tight")
figBstr.savefig(save_path+"\\"+"_BPerpStrengthDifferences.png",bbox_inches="tight")

#%%
"""
Remove Maximum B as Maskj

"""
plt.close("all")
Stack_To_Image = Unwrapped_Phase_Stack
Phi_Crops = []
Bx_Crops = []
By_Crops = []
angle_rots = [40,40,20]
for i in range(len(Stack_To_Image)):
    
    img = Stack_To_Image[i]
    
    img_rot,_ = rotate_image(img,angle_rots[i])
    
    img_crop = img_rot[M[0][0]:M[1][0],M[0][1]:M[1][1]]
    
    Phi_Crops.append(img_crop)
    gradx = np.gradient(img,axis=0)*1/(new_scale*1e-9)
    grady = np.gradient(img,axis=1)*1/(new_scale*1e-9)
    Bx = -hbar/econst*gradx*1E9 #T*nm
    By = hbar/econst*grady*1E9 #T*nm
    
    Bx_rot,_ = rotate_image(Bx,angle_rots[i])
    By_rot,_ = rotate_image(By,angle_rots[i])
    
    Bx_crop = Bx_rot[M[0][0]:M[1][0],M[0][1]:M[1][1]]
    By_crop = By_rot[M[0][0]:M[1][0],M[0][1]:M[1][1]]
    
    Bx_Crops.append(Bx_crop)
    By_Crops.append(By_crop) 
    imfigs = []
    BVal = float(Obj_Data[i].split("_")[-1].split("Obj")[0])/100*2
    if i==0:
        stdX = np.std(Bx_witht)
        w_roi = 200
        vminTot,vmaxTot = np.min(img_crop[540:600,700:800]),np.max(img_crop[540:600,700:800])
    fig,axs = plt.subplots(tight_layout=True,ncols=1,figsize=(10,7))
    imfigs.append(axs.imshow(img_crop,cmap="inferno",vmin=vminTot,vmax=vmaxTot))
       # axs[0].text("$\B_{\perp,x}$",fontsize)
    axs.text(0.01, 0.99, "Applied |B|={:.1f} mT".format(BVal*1E3), horizontalalignment='left',
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
    cbar.ax.set_xlabel("$[T]$",fontsize=20)
    cbar.ax.tick_params(labelsize=12)
    ts = axs.transData
        # rec_rot = plt.Rectangle((c0, r0), c1-c0,r1-r0, fill=False, color="w", ls="dashed", transform=tr + ts)
        
        # axs[n].add_patch(rec_rot)
            # cbar.ax.set_xlim([np.min(Stack_To_Image),np.max(Stack_To_Image)])
#%%

plt.close("all")
Phi0 = Phi_Crops[-1]
Bx0 = Bx_Crops[-1]
By0 = By_Crops[-1]
for i in range(len(Phi_Crops)-1):
    img = Phi_Crops[i]
    img = img - Phi0
    
    img,_ = rotate_image(img,-50)
    
    Bxi = Bx_Crops[i]-Bx0
    Byi = By_Crops[i]-By0
    
    Bxi,_ = rotate_image(Bxi,-50)
    Byi,_ = rotate_image(Byi,-50)
    
    fig,axs = plt.subplots(tight_layout=True,figsize=(8,6))
    axs.imshow(img,cmap="inferno")
    
    points = fig.ginput(n=3)
    points = np.asarray(points)
    
    r0,c0 = int(points[0][1]),int(points[0][0])
    h_roi,w_roi = int(points[2][1]-points[0][1]),int(points[2][0]-points[0][0])
    
    roi_x = img[r0:r0+h_roi,c0:c0+w_roi]
    
    phi_m = np.mean(roi_x,axis=0)
    
    x = np.linspace(0,len(phi_m),len(phi_m))*new_scale
    
    fit = np.polyfit(x,phi_m,1)
    phi_m = phi_m - x*fit[0]
    fig2,axs2 = plt.subplots(tight_layout=True,nrows=2,sharey='row',figsize=(10,8))
    axs2[0].imshow(img,cmap="inferno")
    axs2[1].plot(x,phi_m,"k")
    
    fig3,axs3 = plt.subplots(tight_layout=True,ncols=2,nrows=2,figsize=(10,8))
    axs3[0,0].imshow(Bxi,cmap="inferno")
    axs3[0,1].imshow(Byi,cmap="inferno")
    
    Bxroi=Bxi[r0:r0+h_roi,c0:c0+w_roi]
    Byroi=Byi[r0:r0+h_roi,c0:c0+w_roi]
    
    Bxroi_m = np.mean(Bxroi,axis=0)
    Byroi_m = np.mean(Byroi,axis=0)
    axs3[1,0].plot(x,Bxroi_m,"k")
    axs3[1,1].plot(x,Byroi_m,"k")
    
    for n in range(2):
        rec = plt.Rectangle((c0,r0),w_roi,h_roi,fill=False,color="w",lw=1.5)
        axs3[0,n].add_patch(rec)
        axs3[0,n].axis("off")
        axs3[0,n].text(0.05,0.95,Names_Bs[n],horizontalalignment="left",verticalalignment="top",transform=axs3[0,n].transAxes,
                       color="w",fontsize=20)
    for n in range(2):
        axs3[1,n].set_xlabel("Distance [nm]",fontsize=20)
        axs3[1,n].tick_params(axis="both",labelsize=14)
    axs3[1,0].set_ylabel("$B_\perp \ [T*nm]$",fontsize=20)
    
   #%%
    
"""
Magnetization + Resize

"""
plt.close("all")
Stack_Copy = Unwrapped_Phase_Stack
save_path = r'C:\Users\maslar\OneDrive - Danmarks Tekniske Universitet\MowinShared\Project PhD Mads\MagneticFerrite_Finland\Results_Figs\LocationC_Data\Magnetization'

Magnetization_Stack = []
grad_stack = []
arrows_stack = []
angle_rots = [-15,-15,-15]
for i,Img_cop in enumerate(Stack_Copy):
    Img_cop,_ = rotate_image(Img_cop,angle_rots[i])
    # if i==0:
    img_analyze = Img_cop/(np.max(Img_cop)-np.min(Img_cop))
    _,M = crop_image("Crop",img_analyze - np.min(img_analyze))
    
    roi = Img_cop[M[0][0]:M[1][0],M[0][1]:M[1][1]]
    Cropped_img = roi
    Magnetization_Stack.append(Cropped_img)
    ax, ay = np.gradient(Cropped_img)

# norm = np.sqrt(ax**2 + ay**2)
# ax /= norm
# ay /= norm
    x = np.arange(Cropped_img.shape[0]) * new_scale
    y = np.arange(Cropped_img.shape[1]) * new_scale
    X,Y= np.meshgrid(x,y)
    
    interval = 11
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
    
    # Draw arrows with equal length:
    norm = np.sqrt(arrows[0]**2 + arrows[1]**2)
    # arrows /= norm
    
    # Coordinates of arrows:
    phi_cols,phi_rows  = (X[::interval, ::interval][:-1,:-1] + 0.5 * interval,
                  Y[::interval, ::interval][:-1,:-1] + 0.5 * interval)
    grad_stack.append([phi_cols,phi_rows])
    arrows_stack.append(arrows)
    
    fig,axs = plt.subplots(tight_layout=True)
    axs.imshow(Cropped_img,cmap="gray",origin="upper")
    axs.quiver(phi_cols,phi_rows,arrows[1],-arrows[0],pivot='mid',color="w")
    axs.axis("off")
    scalebar = ScaleBar(new_scale,
                        Reconstructed_Stack[i].axes_manager[0].units, length_fraction=0.5,
                        location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                        )
    axs.add_artist(scalebar)
    save_Name = Obj_Data[i].split("\\")[-1].split(".dm4")[0]
    fig.savefig(save_path+"\\"+save_Name+"Unwrapped_Phase_WithMagnetization.png",bbox_inches="tight")
    plt.close(fig)
    

#%%
# plt.gca().set_aspect('equal')