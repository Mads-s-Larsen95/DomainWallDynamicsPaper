# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:08:18 2024

@author: maslar
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.pyplot import cm
import glob
from scipy.signal import find_peaks
import scipy.optimize as opt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
import scipy.ndimage as ndimage
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

path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\Holography_Location1_ObjecteLensExc_2_SavedFiles\NumpyArrays'

files = glob.glob(path+"\\**.npy")

Unwrapped_Phase_Stack = []
BVal_all = []
scale = 2.0970206260681152
for file in files:

    Unwrapped_Phase_Stack.append(np.load(file))
    BVal = float(file.split("Perc")[0].split("_")[-1])/100*2
    BVal_all.append(BVal)
    
#%%

hbar = 6.626*1E-34/(2*np.pi)
econst = 1.602E-19
front_fact = hbar/econst
font_scalebar = {'family': "Arial",
                 'size': 20,
                 'weight': 'bold'}

scale = 2.0970206260681152
Legend_Labels = ["$\Delta \phi$","$B_{x,\perp}$","$B_{y,\perp}$"]
Stack_To_Analyze = Unwrapped_Phase_Stack
offset = 5
offset_BxBy = 0.0
colors = cm.jet(np.linspace(0,1,len(files)))
plt.close("all")
fig,axs = plt.subplots(tight_layout=True,nrows=3,sharex=True,figsize=(8,10))
for i,img in enumerate(Stack_To_Analyze):
    BVal = BVal_all[i]*1e3
    # img = ndimage.gaussian_filter(img,sigma=2)
    img_rot,rot_mat = rotate_image(img,-220)
    
    rot_mat2 = np.vstack((rot_mat,[0,0,1]))
    inv_rot = np.linalg.inv(rot_mat2)
    tr = mpl.transforms.Affine2D(inv_rot)
    if i==0:
        img_analyze = img_rot/(np.max(img_rot)-np.min(img_rot))
        _,M = crop_image("Crop",img_analyze - np.min(img_analyze))
        c0,c1 = M[0][1], M[1][1]
        r0,r1 = M[0][0], M[1][0]
    
    
    grady = np.gradient(img,axis=0)*1/(scale*1e-9)
    gradx = np.gradient(img,axis=1)*1/(scale*1e-9)
    Bx = -front_fact*grady*1e6
    By = front_fact*gradx*1e6
    
    Bx_rot,_ = rotate_image(Bx,-220)
    By_rot,_ = rotate_image(By,-220)
    
    Bx_vec = np.mean(Bx_rot[r0:r1,c0:c1],axis=0)
    By_vec = np.mean(By_rot[r0:r1,c0:c1],axis=0)
    phi_vec = np.mean(img_rot[r0:r1,c0:c1],axis=0)
    
    if i==0:
        stdX = np.std(Bx)
        vminTot,vmaxTot = np.min(Bx)+3*stdX,np.max(Bx)
        x_vec = np.linspace(0,len(phi_vec),len(phi_vec))*scale
        imfigs = []
        figImgs,axsImgs = plt.subplots(tight_layout=True,ncols=3,figsize=(14,6))
        imfigs.append(axsImgs[0].imshow(img,cmap="inferno"))
        imfigs.append(axsImgs[1].imshow(Bx,cmap="inferno",vmin=vminTot,vmax=vmaxTot))
        imfigs.append(axsImgs[2].imshow(By,cmap="inferno",vmin=vminTot,vmax=vmaxTot))
        
    
        for n in range(3):
            axsImgs[n].text(0.01,0.99,Legend_Labels[n],horizontalalignment="left",verticalalignment="top",transform=axsImgs[n].transAxes,fontsize=20,color="w")
            axsImgs[n].axis("off")
            scalebar = ScaleBar(scale,
                       "nm", length_fraction=0.5,
                       location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                       )
            axsImgs[n].add_artist(scalebar)
            divider = make_axes_locatable(axsImgs[n])
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
            cbar = plt.colorbar(imfigs[n],cax=cax,orientation="horizontal")
            if n==0:
                cbar.ax.set_xlabel("$rad$",fontsize=20)
            else:
                cbar.ax.set_xlabel("$T \cdot  \mu m$",fontsize=20)
            cbar.ax.tick_params(labelsize=12)
            ts = axsImgs[n].transData
            rec_rot = plt.Rectangle((c0, r0), c1-c0,r1-r0, fill=False, color="w", ls="dashed", transform=tr + ts)
            
            axsImgs[n].add_patch(rec_rot)
    
    bck_phi = np.polyfit(x_vec,phi_vec,2)
    
    phi_vec = phi_vec - np.polyval(bck_phi,x_vec)
    if i==0:
        axs[0].plot(x_vec,phi_vec,color=colors[i],label="{:.0f} mT".format(BVal))
        axs[1].plot(x_vec,Bx_vec,color=colors[i])
        axs[2].plot(x_vec,By_vec,color=colors[i])
    if BVal==120:
        axs[0].plot(x_vec,phi_vec+offset,color=colors[i],label="{:.0f} mT".format(BVal))
        axs[1].plot(x_vec,Bx_vec+offset_BxBy,color=colors[i])
        axs[2].plot(x_vec,By_vec+offset_BxBy,color=colors[i])
    if BVal==-120:
        axs[0].plot(x_vec,phi_vec-offset,color=colors[i],label="{:.0f} mT".format(BVal))
        axs[1].plot(x_vec,Bx_vec-offset_BxBy,color=colors[i])
        axs[2].plot(x_vec,By_vec-offset_BxBy,color=colors[i])
for n in range(3):
    axs[n].set_xlim([x_vec[0],x_vec[-1]])
    axs[n].tick_params(axis="both",labelsize=14)
    axs[n].text(0.01,0.99,Legend_Labels[n],verticalalignment="top",horizontalalignment="left",transform=axs[n].transAxes,fontsize=20)
axs[0].set_ylabel("$\Delta \phi \ [rad]$",fontsize=20)
handles,labels = axs[0].get_legend_handles_labels()
axs[0].legend(handles=[handles[1],handles[0],handles[2]],loc="upper left",bbox_to_anchor=(1.05,1),fontsize=14)
axs[1].set_ylabel("$B_{x,\perp} \ [T*\mu m]$",fontsize=20)
axs[2].set_ylabel("$B_{y,\perp} \ [T*\mu m]$",fontsize=20)
axs[2].set_xlabel("Distance [nm]",fontsize=20)