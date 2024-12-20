
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:50:26 2024

@author: maslar
"""

import numpy as np
from matplotlib import pyplot as plt
import glob
import hyperspy.api as hs
import skimage
import cv2
import scipy
from scipy import ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib as mpl
import os
from scipy.signal import find_peaks
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
#%%
plt.close("all")
vacc = 300000
lmbda = 12.26e-10 / np.sqrt(vacc + 0.9788e-6 * vacc**2)


# Images should be aligned and of the same width and height in pixels
path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\TIE_location1'
files = glob.glob(path+"\\**.dm4")

start = 50
infocus_filename = files[0]
underfoc_files = [files[1],files[4],files[5],files[8],files[9],files[12],files[13],files[16],files[17],files[20]]
overfoc_files = [files[2],files[3],files[6],files[7],files[10],files[11],files[14],files[15],files[18],files[19]]

print(len(underfoc_files),len(overfoc_files))

     
resize_fact = 1/3

z_list = np.array([10,20,30,50,80,130,180,230,300,400])*1e-6 #in m

infocus_data = hs.load(infocus_filename)
gx = infocus_data.axes_manager["x"].scale  # grid (pixel) size in meters
gy = infocus_data.axes_manager["y"].scale
cal_factor = 0.0289415
if infocus_data.axes_manager["x"].units == "nm":
    gx,gy = gx*1e-9,gy*1e-9
else:
    gx,gy = gx*1e-6,gy*1e-6

gx,gy = gx/resize_fact,gy/resize_fact
img_infocus = infocus_data.data*cal_factor
img_infocus = np.where(img_infocus == 0, 0.001, img_infocus)
img_infocus_res = skimage.transform.resize(img_infocus,(int(img_infocus.shape[0]*resize_fact),int(img_infocus.shape[1]*resize_fact)))
img_analyze = img_infocus_res/(np.max(img_infocus_res)-np.min(img_infocus_res))
img_analyze = img_analyze - np.min(img_analyze)
_,M_temp = crop_image("Crop",img_analyze)
template = img_infocus_res[M_temp[0][0]:M_temp[1][0],M_temp[0][1]:M_temp[1][1]]


fig,axs = plt.subplots(tight_layout=True)
axs.imshow(template)
axs.axis("off")
h,w = np.shape(template)
res = cv2.matchTemplate(img_infocus_res,template,cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left0 = max_loc

# _,M_roi = crop_image("Crop",img_analyze)
M_roi = M_temp
img_infocus_roi = img_infocus_res[M_roi[0][0]:M_roi[1][0],M_roi[0][1]:M_roi[1][1]]

fig,axs = plt.subplots(tight_layout=True,ncols=2)
axs[0].imshow(template)
axs[1].imshow(img_infocus_res)
rec = plt.Rectangle(top_left0,w,h,fill=False,color="w")
axs[1].add_patch(rec)
for n in range(2):
    axs[n].axis("off")
axs[0].set_title("Template")
axs[1].set_title("Matched Template")

underfoc_overfoc_imgs = []
underfoc_overfoc_imgs_rois = []
count = 0
save_path_imgs_tmp = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\InLine_EH\Template_Matching\Location_1'
save_path_NP_tmp = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\InLine_EH\Template_Matching\Location_1'
save_files = 1
gxi = gx*1e9
font_scalebar = {'family': "Arial",
                 'size': 30,
                 'weight': 'bold'}

for i in range(len(overfoc_files)):
    fig,axs = plt.subplots(tight_layout=True,ncols=3,nrows=2,figsize=(15,15))
    count += 1
    underfoc_filename_i = underfoc_files[i]
    overfoc_filename_i = overfoc_files[i]
    
    print([underfoc_filename_i.split("\\")[-1],overfoc_filename_i.split("\\")[-1],int(z_list[i]*1e6)])
    underfocus_data = hs.load(underfoc_filename_i)
    overfocus_data = hs.load(overfoc_filename_i)

    img_underfocus = underfocus_data.data*cal_factor
    # img_underfocus = np.where(img_underfocus == 0, 0.001, img_underfocus)
    img_overfocus = overfocus_data.data*cal_factor
    # img_overfocus = np.where(img_overfocus == 0, 0.001, img_overfocus)


    images = [img_underfocus,img_overfocus]
    images_new = []
    rois = []
    for j,im in enumerate(images):
        
        img_resized = skimage.transform.resize(im,(int(im.shape[0]*resize_fact),int(im.shape[1]*resize_fact)))
        res = cv2.matchTemplate(img_resized,template,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        axs[j,0].imshow(img_resized,cmap="gray")
        scalebar = ScaleBar(gxi,
                            "nm", length_fraction=0.5,
                            location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                            )
        axs[j,0].add_artist(scalebar)
        rec = plt.Rectangle(top_left,w,h,fill=False,color="w")
        axs[j,0].add_patch(rec)
        axs[j,0].set_xticks([])
        axs[j,0].set_yticks([])
        
        
        tx = top_left0[0] - top_left[0]
        ty = top_left0[1] - top_left[1]
        
        M_trans = np.float64([
                            [1,0,tx],
                            [0,1,ty]])
        shifted = cv2.warpAffine(img_resized, M_trans, (img_resized.shape[1], img_resized.shape[0]))
        
        axs[j,1].imshow(shifted,cmap="gray")
        rec = plt.Rectangle(top_left0,w,h,fill=False,color="w")
        axs[j,1].add_patch(rec)
        axs[j,1].set_xticks([])
        axs[j,1].set_yticks([])
        scalebar = ScaleBar(gxi,
                            "nm", length_fraction=0.5,
                            location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                            )
        axs[j,1].add_artist(scalebar)
        
        roi = shifted[M_roi[0][0]:M_roi[1][0],M_roi[0][1]:M_roi[1][1]]
        rois.append(roi)
        axs[j,2].imshow(roi,cmap="gray")
        axs[j,2].set_xticks([])
        axs[j,2].set_yticks([])
        scalebar = ScaleBar(gxi,
                            "nm", length_fraction=0.5,
                            location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                            )
        axs[j,2].add_artist(scalebar)
        images_new.append(img_resized)
    axs[0,0].set_title("Find Template",fontsize=30)
    axs[0,0].set_ylabel("-f",fontsize=30)
    axs[1,0].set_ylabel("+f",fontsize=30)
    axs[0,1].set_title("Translate Image",fontsize=30)
    axs[0,2].set_title("ROI",fontsize=30)
    save_name = "ROI_{:.0f}__um".format(z_list[i]*1e6)
        
        
    if save_files:
        np.save(save_path_NP_tmp+"\\"+save_name+".npy",rois)
        fig.savefig(save_path_imgs_tmp+"\\"+save_name+".png",bbox_inches="tight")
    plt.close(fig)
    
    # diff = rois[1]-rois[0]
    # # fig2,axs2 = plt.subplots(tight_layout=True,ncols=3)
    # axs2.axis("off")
    # axs2[1].imshow()
    # axs2[2].imshow(diff)
    underfoc_overfoc_imgs.append(images_new)
    underfoc_overfoc_imgs_rois.append(rois)
    # if count==2:
    #     break
underfoc_overfoc_imgs = np.asarray(underfoc_overfoc_imgs)

#%%

def inverse_laplacian(img, gx, gy, zz,apply_mask = False,save_files = True):
    fft_f = np.fft.fftshift(np.fft.fft2(img))
    gxi = gx*1e9
    if save_files:
        save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\InLine_EH\FFTs\Location_1'
        
        fig,axs = plt.subplots(tight_layout=True,ncols=4,figsize=(18,6))
        axs[0].imshow(np.log(abs(fft_f)),cmap="gray")
        axs[0].set_title(r"$FFT(I(\mathbf{r}))$",fontsize=20)
        
    my,mx = fft_f.shape
    kx,ky = np.fft.fftfreq(mx,gx),np.fft.fftfreq(my,gy)
    KX,KY = np.meshgrid(kx,ky,indexing="xy")
    
    Q = np.sqrt(KY**2 + KX**2)
    
    kf = 1/Q**2
    
    kf = np.fft.fftshift(kf)
    kf[int(my//2),int(mx//2)] = 0
    
    fft_f *= kf#*fft_f #kf*F
    front_fact = -1/((2*np.pi)**2) 
    inv_FFT = front_fact * np.fft.ifft2(np.fft.ifftshift((fft_f))) #FT^-1(kf*F)
    if save_files:
        
        axs[1].imshow(np.log(abs(kf)),cmap="gray")
        axs[1].set_title(r"$FFT(\nabla^{-2})$",fontsize=20)
        axs[2].imshow(np.log(abs(fft_f)),cmap="gray")
        axs[2].set_title(r"$FFT(\nabla^{-2} * I(\mathbf{r}))$",fontsize=20)
        imfig = axs[3].imshow(np.real(inv_FFT),cmap="gray")
        axs[3].set_title(r"$FFT^{-1}(FFT(\nabla^{-2} * I(\mathbf{r})))$",fontsize=20)
        for n in range(3):
            scalebar = ScaleBar(1/(gxi),
                            "1/nm", length_fraction=0.5,dimension='si-length-reciprocal',
                            location='lower right', box_alpha=0, color='b',font_properties = font_scalebar
                            )
            axs[n].add_artist(scalebar)
        
        scalebar = ScaleBar(gxi,
                            "nm", length_fraction=0.5,
                            location='lower right', box_alpha=0, color='b',font_properties = font_scalebar
                            )
        axs[-1].add_artist(scalebar)
        
        divider = make_axes_locatable(axs[-1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(imfig,cax=cax,orientation="vertical")
        # cbar.ax.set_xlabel("$rad$",fontsize=20)
        cbar.ax.tick_params(labelsize=12)
        for n in range(4):
            axs[n].axis("off")
        fig.savefig(save_path+"\\FFTPlots_{:.0f}_um.png".format(zz*multiplier),bbox_inches="tight")
        plt.close(fig)

    
    return inv_FFT

def calculate_TIE(img_overfoc_i,img_underfoc_i,img_infocus_i,gx,gy,focal_val,sigma_blur = 3,apply_mask=False,save_files = True):
    invdiv = (img_overfoc_i - img_underfoc_i) 
    invdiv2 = invdiv/img_infocus_i
    
    invdiv2 = ndimage.gaussian_filter(invdiv2,sigma=sigma_blur)
    if save_files:
        save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\InLine_EH\Difference\Location_1'
        imfigs = []
        fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(12,6))
        imfigs.append(axs[0].imshow(invdiv,cmap="gray"))
        axs[0].set_title(r"$I(\mathbf{r},+f) - I(\mathbf{r},-f)$",fontsize=20)
        imfigs.append(axs[1].imshow(invdiv2,cmap="gray"))
        axs[1].set_title(r"$I(\mathbf{r},+f) - I(\mathbf{r},-f)/I(\mathbf{r},0)$",fontsize=20)
        for n in range(2):
            scalebar = ScaleBar(gx*1e9,
                            "nm", length_fraction=0.5,
                            location='lower right', box_alpha=0, color='b',font_properties = font_scalebar
                            )
            axs[n].add_artist(scalebar)
            axs[n].axis("off")
            divider = make_axes_locatable(axs[n])
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
            cbar = plt.colorbar(imfigs[n],cax=cax,orientation="horizontal")
            # cbar.ax.set_xlabel("$rad$",fontsize=20)
            cbar.ax.tick_params(labelsize=12)
        fig.savefig(save_path+"\\"+"DifferencePlots_{:.0f}_um.png".format(focal_val*multiplier),bbox_inches="tight")
        plt.close(fig)
        
    # print(
    # qi[0,0] = 1
    inv_laplacian = inverse_laplacian(invdiv2,gx,gy,focal_val,apply_mask,save_files = save_files)
    
    # inv_laplacian = np.real(inv_laplacian)
    
    phi_ = -np.pi/(lmbda*focal_val)*inv_laplacian
    
    phi = -np.real(phi_)

    return phi


def magnetization(phi_i,gx,gy):
    grad_y = np.gradient(phi_i,axis=0)#scipy.signal.convolve2d(phi_i, kernel_y, mode="same", boundary="wrap")
    grad_x =  np.gradient(phi_i,axis=1)#scipy.signal.convolve2d(phi_i, kernel_x, mode="same", boundary="wrap")
    front_fact = scipy.constants.hbar/(scipy.constants.e) #T*m^2
    # front_fact = 1
    Bx = front_fact*grad_y*1/gx #T*m
    # print(np.shape(Bx))
    By = -front_fact*grad_x*1/gy #T*m
    B = np.array([Bx,By])
    # print(np.shape(B))
    return B

def fft_and_plot_1d(f,gx,h_roi,show_imgs=True):
    
    
    # r0,c0 
    fft_f = np.fft.fftshift(np.fft.fft2(f))
    
    # fft_f = fft_f.astype(np.float32)
    
    
    fft_f = np.log(abs(fft_f))
    fft_scale = 1/gx
    
    
    value = h_roi//2#np.sqrt(((fft_f.shape[0]/2.0)**2.0)+((fft_f.shape[1]/2.0)**2.0))
    
    print(fft_f.shape)
    polar_fft = cv2.linearPolar(fft_f,(h_roi/2, h_roi/2), value, cv2.WARP_FILL_OUTLIERS)
    if show_imgs:
        fig,axs = plt.subplots(tight_layout=True,ncols=3)
        axs[0].imshow(f,cmap="gray")
        rec = plt.Rectangle((M[0][1],M[0][0]),h_roi,h_roi,fill=False,color="r")
        axs[0].add_patch(rec)
        
        axs[1].imshow(np.log(abs(fft_f)),cmap="gray")
        axs[2].imshow(np.log(abs(polar_fft)),cmap="gray")
        for n in range(3):
            axs[n].axis("off")
    mean_polar = np.mean(polar_fft[2:-2,:],axis=0)
    
    x_polar = np.linspace(0,len(mean_polar),len(mean_polar))*fft_scale
    return mean_polar,x_polar

def chi(k,lmbda,Cs,df):
    k0 = 2*np.pi/lmbda
    chi = k0*(lmbda**2*k**2/2 * df + lmbda**4 * k**4/4 * Cs)
    return chi

#%%
if np.mean(z_list)<1e-6:
    multiplier = 1e9
    unit_z = "nm"
elif np.mean(z_list)>1e-6:
    multiplier = 1e6
    unit_z = "um"
vacc = 300000
lmbda = 12.26e-10 / np.sqrt(vacc + 0.9788e-6 * vacc**2) #m
plt.close("all")
# Stack_Copy = underfoc_overfoc_imgs
fig,axs = plt.subplots(tight_layout=True)
fig2,axs2 = plt.subplots(tight_layout=True)
fig3,axs3 = plt.subplots(tight_layout=True)
defoc_vals = []
for i in range(len(underfoc_overfoc_imgs)-5):
    
    img_underfoc = underfoc_overfoc_imgs[i+5][0]
    # img_overfoc = underfoc_overfoc_imgs[i][1]
    
    if i==0:
        img_analyze = img_infocus_res/(np.max(img_infocus_res)-np.min(img_infocus_res))
        img_analyze = img_analyze - np.min(img_analyze)
        _,M = crop_image("Crop",img_analyze)
        h_roi = int(M[1][0] - M[0][0])
    img_underfoc_crop = img_underfoc[M[0][0]:M[0][0]+h_roi,M[0][1]:M[0][1]+h_roi]
    # mean_plus,x_plus = fft_and_plot_1d(img_overfoc,gx)
    mean_min,x_min = fft_and_plot_1d(img_underfoc_crop,gx,h_roi)
    
    N = 5
    mean_min = np.convolve(mean_min, np.ones(N)/N, mode='valid')
    x_min = np.convolve(x_min, np.ones(N)/N, mode='valid')
    # axs.plot(x_plus,mean_plus,"k")
    axs.plot(x_min,mean_min,"k")
    valleys,_ = find_peaks(1/mean_min,prominence=0.005)
    valleys = valleys[1:]
    peaks,_ = find_peaks(mean_min,prominence=0.2)
    # peaks = peaks[1:]
    axs.plot(x_min[valleys],mean_min[valleys],"gx")
    axs.plot(x_min[peaks],mean_min[peaks],"yx")
    
    valleys_peaks = sorted(np.concatenate([valleys,peaks]))
    
    n2 = np.arange(2,(len(valleys_peaks)+2),1)
    
    x_valleyspeaks = x_min[valleys_peaks]
    
    x_valleyspeaks2 = x_valleyspeaks**2
    
    y = n2/x_valleyspeaks2
    
    x = x_valleyspeaks2
    
    x,y = x[1:],y[1:]
    
    axs2.plot(x,y,"o")
    
    polfit = np.polyfit(x,y,1)
    axs2.plot(x,np.polyval(polfit,x),"--")
    
    intercept = polfit[-1]
    
    defoc = intercept/(2*lmbda)
        
    defoc_vals.append(defoc)
    
    
    axs3.plot(z_list[i]*multiplier,defoc*multiplier)
        
        # y = 
# axs.set_xscale("log")
# axs.set_ys
# cale("log")
        
#%%
plt.close("all")
vacc = 300000
lmbda = 12.26e-10 / np.sqrt(vacc + 0.9788e-6 * vacc**2) #m

z_list = np.array([10,20,30,50,80,130,180,230,300,400]) * 1e-6#um*1e-6 #in m
if np.mean(z_list)<1e-6:
    multiplier = 1e9
    unit_z = "nm"
elif np.mean(z_list)>1e-6:
    multiplier = 1e6
    unit_z = "um"
gx = infocus_data.axes_manager["x"].scale #/ resize_fact # grid (pixel) size in meters
gy = infocus_data.axes_manager["y"].scale #/ resize_fact 
if infocus_data.axes_manager["x"].units == "nm":
    gx,gy = gx*1e-9,gy*1e-9
else:
    gx,gy = gx*1e-6,gy*1e-6
gx,gy = gx/resize_fact, gy/resize_fact
# gx,gy = gx*1e6,gy*1e6
# lmbda=lmbda*1e6
# lmbda = 1.96e-12

font_scalebar = {'family': "Arial",
                 'size': 30,
                 'weight': 'bold'}
# plt.close("all")
Phase_Stack_TIE = []
B_all = []
save_path_1 = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\InLine_EH\Phase\Location_1'
save_path_NP = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\InLine_EH\Phase\Location_1'
save_files_1 = 1

for i in range(len(underfoc_overfoc_imgs_rois)):
    zz = z_list[i]
    print(zz*multiplier)
    img_underfoc = underfoc_overfoc_imgs_rois[i][0]
    img_overfoc = underfoc_overfoc_imgs_rois[i][1]
    imfigs = []
    
    if i==len(underfoc_overfoc_imgs_rois)-1:
        fig,axs = plt.subplots(tight_layout=True,ncols=2)
        axs[0].imshow(img_underfoc,cmap="gray")
        axs[1].imshow(img_overfoc,cmap="gray")
        break
img_infoc_i = img_infocus_roi
for i in range(len(underfoc_overfoc_imgs_rois)):
    zz = z_list[i]
    print(zz*multiplier)
    img_underfoc_i = underfoc_overfoc_imgs_rois[i][0]
    img_overfoc_i = underfoc_overfoc_imgs_rois[i][1]
    imfigs = []
    
    # if i==0:
    #     fig,axs = plt.subplots(tight_layout=True)
    #     axs.imshow(img_underfoc,cmap="gray")
    #     break
    phi = calculate_TIE(img_overfoc_i, img_underfoc_i, img_infoc_i,gx,gy, zz,sigma_blur = 1,apply_mask = False,save_files = save_files_1)
    # phi = TIE2(img_overfoc, img_underfoc, img_infocus_roi, gx,gy,zz)
    fig,axs = plt.subplots(tight_layout=True,ncols=2,figsize=(10,6))
    imfigs.append(axs[0].imshow(img_infocus_roi,cmap="gray"))
    imfigs.append(axs[1].imshow(phi,cmap="gray"))
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    cbar = plt.colorbar(imfigs[1],cax=cax,orientation="horizontal")
    cbar.ax.set_xlabel("$ \phi \ [rad]$",fontsize=20)
    cbar.ax.tick_params(labelsize=12)
    for n in range(2):
        axs[n].axis("off")
        scalebar = ScaleBar(gx*1e9,
                            "nm", length_fraction=0.5,
                            location='lower right', box_alpha=0, color='b',font_properties = font_scalebar
                            )
        axs[n].add_artist(scalebar)
    if save_files_1 == 1:
        fig.savefig(save_path_1+"\\"+"TIE_Phase_{:.0f}_um.png".format(zz*multiplier),bbox_inches="tight")
        np.save(save_path_NP+"\\"+"TIE_Phase_{:.0f}_um.npy".format(zz*multiplier),phi)
        np.save(save_path_NP+"\\"+"Scale_Resized_m_per_px.npy",gx)
        plt.close(fig)
    Phase_Stack_TIE.append(phi)
    B = magnetization(phi,gx,gy)
    B_all.append(B)
    
#%%
plt.close("all")
i = 6
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images'
n_cols,n_rows = 3,2
focal_val = z_list[i]
img_underfoc = underfoc_overfoc_imgs_rois[i][0]
img_overfoc = underfoc_overfoc_imgs_rois[i][1]

invdiv = (img_underfoc - img_overfoc) #/ (2*focal_val)
invdiv2 = 1/img_infocus_roi * invdiv
inv_laplacian = inverse_laplacian(invdiv2,gx,gy,focal_val,apply_mask=False,save_files = 0)
front_fact = -2*np.pi/(lmbda*((2*focal_val)))
phi_ = front_fact*inv_laplacian
    
phi = np.real(phi_)
imfigs = []
fig,axs = plt.subplots(tight_layout=True,ncols=n_cols,nrows=n_rows,figsize=(16,10))
imfigs.append(axs[0,0].imshow(img_underfoc,cmap="gray"))
imfigs.append(axs[1,0].imshow(img_overfoc,cmap="gray"))
imfigs.append(axs[0,1].imshow(invdiv,cmap="gray"))
imfigs.append(axs[1,1].imshow(invdiv2,cmap="gray"))
imfigs.append(axs[0,2].imshow(np.real(inv_laplacian),cmap="gray"))
imfigs.append(axs[1,2].imshow(phi,cmap="gray"))

lbls = ["(a)","(b)","(c)","(d)","(e)","(f)"]
count = 0

for nc in range(n_cols):
    for nr in range(n_rows):
        axs[nr,nc].axis("off")
        axs[nr,nc].text(0.01,0.99,lbls[count],fontsize=30,horizontalalignment="left",
                        verticalalignment="top",fontname="Arial",color="w")
        divider = make_axes_locatable(axs[nr,nc])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # print(nc+nr)
        scalebar = ScaleBar(gx*1e9,
                            "nm", length_fraction=0.5,
                            location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                            )
        axs[nr,nc].add_artist(scalebar)
        print(count)
        
        cbar = plt.colorbar(imfigs[count],cax=cax,orientation="vertical")
        # cbar.ax.set_xlabel("$ \phi \ [rad]$",fontsize=20)
        cbar.ax.tick_params(labelsize=12)   
        count+=1
fig.savefig(save_path+"\\"+"Inline_ImageAnalysis_{:.0f}.png".format(focal_val*1e6),bbox_inches="tight")
#%%
plt.close("all")
Stack_To_Analyze = Phase_Stack_TIE
angle_rots = 140
j = 6
rotimg,rot_mat = rotate_image(underfoc_overfoc_imgs_rois[j][1],angle_rots)
rot_mat2 = np.vstack((rot_mat,[0,0,1]))
inv_rot = np.linalg.inv(rot_mat2)
tr = mpl.transforms.Affine2D(inv_rot)
img_analyze = rotimg/(np.max(rotimg)-np.min(rotimg))
_,mat = crop_image("Crop",img_analyze - np.min(img_analyze))
c0,c1 = mat[0][1], mat[1][1]
r0,r1 = mat[0][0], mat[1][0]
fig,axs = plt.subplots(tight_layout=True,figsize=(8,6))
figImg,axsImg = plt.subplots(tight_layout=True,figsize=(8,6))
Cropped_Imgs = []
phi_mat_tie = []

save_path_NP = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Files_NP\InLine_EH\Linescan\Location_1'
save_path = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\Saved_Images\InLine_EH\Linescan\Location_1'
for i,img in enumerate(Stack_To_Analyze):
    rotimg,_ = rotate_image(img,angle_rots)
    
    # axs[0].imshow(rotimg,cmap="inferno")
    # axs[1].imshow(rotimg,cmap="inferno")
    roi = rotimg[M[0][0]:M[1][0],M[0][1]:M[1][1]]
    Cropped_img = roi
    phi_vec_tie = np.mean(roi,axis=0) 
    # phi_vec_tie = phi_vec_tie - phi_vec_tie[0]
    
    np.save(save_path_NP+"\\"+"TIE_Phase_{:.0f}_um.npy".format(z_list[i]*multiplier),phi_vec_tie)
    np.save(save_path_NP+"\\"+"Scale_Resized_m_per_px.npy",gx)
    if i==0:
        x_tie = np.linspace(0,len(phi_vec_tie),len(phi_vec_tie))*gx*1e9
    
    poly = np.polyfit(x_tie,phi_vec_tie,2)
    if i==j:
        imfig = axsImg.imshow(Stack_To_Analyze[j],cmap="gray")
        ts = axsImg.transData
        rec_rot = plt.Rectangle((c0, r0), c1-c0,r1-r0, 
                                fill=False, color="w", ls="dashed", transform=tr + ts)
        axsImg.arrow((c1-c0)/2+c0-50/2, (r1-r0)/2+r0, 50, 0, width = 2,color="w",transform=tr+ts)
        axsImg.add_patch(rec_rot)
        scalebar = ScaleBar(gx*1e9,
                                    "nm", length_fraction=0.5,
                                    location='lower right', box_alpha=0, color='w',font_properties = font_scalebar
                                    )
        axsImg.add_artist(scalebar)
        divider = make_axes_locatable(axsImg)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        cbar = plt.colorbar(imfig,cax=cax,orientation="vertical")
        cbar.ax.set_ylabel("$ \phi \ [rad]$",fontsize=20)
        cbar.ax.tick_params(labelsize=12)   
        axsImg.axis("off")
        
        figImg.savefig(save_path+"\\"+"TIE_Phase_{:.0f}_um.png".format(zz*multiplier))#,bbox_inches="tight")

    phi_vec_tie = phi_vec_tie - np.polyval(poly,x_tie)

    axs.plot(x_tie,phi_vec_tie,label="f={:.0f}".format(z_list[i]*multiplier) + unit_z)
    
    
    phi_mat_tie.append(phi_vec_tie)
axs.set_xlabel("Distance [nm]",fontsize=20)
axs.set_ylabel("$\phi \ [rad]$",fontsize=20)
axs.tick_params(axis="both",labelsize=14)
axs.legend(loc="upper left",bbox_to_anchor=(1.05,1),fontsize=12)

#%%
save_folder = "Phase_NPY"
save_path = path + "\\" + save_folder
if not os.path.exists(save_path):
    os.makedirs(save_path)
fig,axs = plt.subplots(tight_layout=True,figsize=(8,6))
for i,phi in enumerate(phi_mat_tie):
    if i > 4:
        axs.plot(x_tie,phi,label="f={:.0f}".format(z_list[i]*multiplier) + unit_z)
        np.save(save_path+"\\"+"TIE_Phase_{:.0f}_um.npy".format(z_list[i]*multiplier),phi)
    if i==0:
        np.save(save_path+"\\"+"x_TIE_Phase_{:.0f}_um.npy".format(z_list[i]*multiplier),x_tie)
axs.set_xlabel("Distance [nm]",fontsize=20)
axs.set_ylabel("$\phi \ [rad]$",fontsize=20)
axs.tick_params(axis="both",labelsize=14)
axs.legend(loc="upper left",bbox_to_anchor=(1.05,1),fontsize=12)       
#%% 
#compare to holo
plt.close("all")
path_holo = r'\\ait-pdfs.win.dtu.dk\Services\NLAB\cen-archive\P87500-Murat\12032024_Mari_Samples\sampleE\TIE_location1\From_Holo'
holo_phi = np.load(path_holo+"\\0BMagnetization_phase.npy")
holo_x = np.load(path_holo + "\\0BMagnetization_x.npy")
fig,axs = plt.subplots(tight_layout=True)
axs.plot(x_tie,phi_mat_tie[j],color="k",label="TIE $\phi (f={:.0f}$".format(z_list[j]*multiplier) + unit_z+")",lw=2)
holo_phi = holo_phi-holo_phi[0]
poly_holo = np.polyfit(holo_x,holo_phi,2)
holo_phi = holo_phi - np.polyval(poly_holo,holo_x)
axs.plot(holo_x,holo_phi,color="r",label="Holo $\phi$",lw=2)
axs.set_xlabel("Distance [nm]",fontsize=20)
axs.set_ylabel("$\phi \ [rad]$",fontsize=20)
axs.tick_params(axis="both",labelsize=14)
axs.legend(loc="upper left",fontsize=14)



#%%
# To show codes
def inverse_laplacian(img, gx, gy):
    fft_f = np.fft.fftshift(np.fft.fft2(img))

    my,mx = fft_f.shape
    kx,ky = np.fft.fftfreq(mx,gx),np.fft.fftfreq(my,gy)
    KX,KY = np.meshgrid(kx,ky,indexing="xy")
    
    Q = np.sqrt(KY**2 + KX**2)
    
    kf = 1/Q**2
    
    kf = np.fft.fftshift(kf)
    kf[int(my//2),int(mx//2)] = 0
    
    fft_f *= kf#*fft_f #kf*F
    
    front_fact = -1/((2*np.pi)**2)
    inv_FFT =  front_fact * np.fft.ifft2(np.fft.ifftshift((fft_f))) #FT^-1(kf*F)

    return inv_FFT

def calculate_TIE(img_overfoc_i,img_underfoc_i,img_infocus_i,gx,gy,focal_val):
    invdiv = (img_overfoc_i - img_underfoc_i) 
    invdiv2 = invdiv/img_infocus_i
    
    inv_laplacian = inverse_laplacian(invdiv2,gx,gy)
    
    phi_ = -np.pi/(lmbda*focal_val)*inv_laplacian
    
    phi = np.real(phi_)

    return phi




