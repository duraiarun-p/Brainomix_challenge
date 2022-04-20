#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 09:40:29 2022
Code for Brainomix Challenge
@author: arun
"""

#%% Library importing
import os
from os import listdir
from os.path import isfile, join
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sp
import scipy.ndimage
from skimage.feature import greycomatrix, greycoprops
from sklearn.cluster import KMeans
#%% Function Definitions
# Reading Files
def read_nii_img_file(inputImageFileName):
    image=sitk.ReadImage(inputImageFileName)
    imgA=sitk.GetArrayFromImage(image)
    return image,imgA
# MATLAB imfill equivalent function
#Source: https://stackoverflow.com/questions/36294025/python-equivalent-to-matlab-funciton-imfill-for-grayscale
def flood_fill(test_array,h_max=255):
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array
# MATLAB imcomplement equivalent function
# Reference: https://ojskrede.github.io/inf4300/exercises/week_11/
def imcomplement(image):
  min_type_val = image.min()
  max_type_val = image.max()
  return min_type_val + max_type_val - image
#
def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))
#%% First Function
def lunganalysis_1(V):  
    siz=np.shape(V)
    Vbodymask=np.zeros(siz,dtype='int16')
    Vbody=np.zeros(siz,dtype='int16')
    Vlungsmask=np.zeros(siz,dtype='int16')
    Vlung=np.zeros(siz,dtype='int16')
    # Vlungvol=np.zeros((siz[0],1),dtype='int16')
    # Vvesselvol=np.zeros((siz[0],1),dtype='int16')
    Vvesselmask=np.zeros(siz,dtype='int16')
    #Thresholding to create mask. The values are chosen after performing histogram 
    #analysis for all the scans.
    Vmin=np.min(V)
    LungThresh=Vmin+865
    BodyThresh=Vmin+265
    VesselThresh=Vmin+924    
    # i=1
    G1_c=[]
    G2_d=[]
    G3_h=[]
    G4_e=[]
    G5_cr=[]
    for i in range(siz[0]):
        sl=V[i,:,:]#slice wise operation
        ##Body Segmentation
        sl1=np.float32(sl<BodyThresh)
        #kernel definiion at each Morpholigcal operation is mandate for this code
        kernel=np.ones((8,8),np.uint8)
        sl1=cv2.morphologyEx(sl1, cv2.MORPH_CLOSE, kernel)
        sl1=flood_fill(sl1)
        sl1=imcomplement(sl1)
        sl1=flood_fill(sl1) 
        sl1=sl1.astype('int16')
        Vbodymask[i,:,:]=sl1#Body mask volume
        sl_n_1 = cv2.multiply(sl, sl1)#Masks is applied
        Vbody[i,:,:]=sl_n_1#Body scan volume
        ##Lung Segmentation
        sl2=np.float32(sl_n_1<LungThresh)
        kernel=np.ones((8,8),np.uint8)
        sl2=cv2.morphologyEx(sl2, cv2.MORPH_CLOSE, kernel)
        sl2=flood_fill(sl2)
        kernel=np.ones((7,7),np.uint8)
        sl2=cv2.erode(sl2,kernel,iterations = 2)#Lung mask
        sl2=sl2.astype('int16')
        Vlungsmask[i,:,:]=sl2#Lung mask volume
        sl_n_2=cv2.multiply(sl_n_1,sl2)#Lung mask applied
        Vlung[i,:,:]=sl_n_2#Lung scan volume
        ## Routine for analysis volume and possibly texture
        sl_n_2_t=sl_n_2
        sl_n_2_t = (1*(sl_n_2_t - np.min(sl_n_2_t))/np.ptp(sl_n_2_t)).astype(int)
        # sl_n_2_t = 255 * sl_n_2_t
        Glcmatrix = greycomatrix(sl_n_2_t, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],levels=4)
        G_1_contrast=greycoprops(Glcmatrix, prop='contrast')
        G_2_dissimilarity=greycoprops(Glcmatrix, prop='dissimilarity')
        G_3_homogeneity=greycoprops(Glcmatrix, prop='homogeneity')
        G_4_energy=greycoprops(Glcmatrix, prop='energy')
        G_5_correlation=greycoprops(Glcmatrix, prop='correlation')
        G1_c.append(np.mean(G_1_contrast))
        G2_d.append(np.mean(G_2_dissimilarity))
        G3_h.append(np.mean(G_3_homogeneity))
        G4_e.append(np.mean(G_4_energy))
        G5_cr.append(np.mean(G_5_correlation))
        # Vlungvol[i]=np.count_nonzero(sl2)
        ##Vessel Segmentation
        sl31=sl_n_2!=0
        sl32=sl_n_2>VesselThresh
        sl31=sl31.astype('int16')
        sl32=sl32.astype('int16')
        sl3=cv2.multiply(sl31,sl32)
        ## Routine for analysis-volume
        # Vvesselvol[i]=np.count_nonzero(sl3)
        Vvesselmask[i,:,:]=sl3
    Vlungvol=np.count_nonzero(Vlungsmask)
    Vvesselvol=np.count_nonzero(Vvesselmask)
    G1_c=sum(G1_c)
    G2_d=sum(G2_d)
    G3_h=sum(G3_h)
    G4_e=sum(G4_e)
    G5_cr=sum(G5_cr)
    ves_lun_rat=(Vvesselvol/Vlungvol)*100
    return Vbody,Vbodymask,Vlung,Vlungsmask,Vvesselmask,Vlungvol,Vvesselvol,G1_c,G2_d,G3_h,G4_e,G5_cr,ves_lun_rat

#%% Second Function
def lunganalysis_2(V):  
    siz=np.shape(V)
    Vbodymask=np.zeros(siz,dtype='int16')
    Vbody=np.zeros(siz,dtype='int16')
    Vlungsmask=np.zeros(siz,dtype='int16')
    Vlung=np.zeros(siz,dtype='int16')
    # Vlungvol=np.zeros((siz[0],1),dtype='int16')
    # Vvesselvol=np.zeros((siz[0],1),dtype='int16')
    Vvesselmask=np.zeros(siz,dtype='int16')
    #Thresholding to create mask. The values are chosen after performing histogram 
    #analysis for all the scans.
    Vmin=np.min(V)
    LungThresh=Vmin+2724
    BodyThresh=Vmin+2524
    VesselThresh=Vmin+2324    
    # i=1
    G1_c=[]
    G2_d=[]
    G3_h=[]
    G4_e=[]
    G5_cr=[]

    siz=np.shape(V)
    Vbodymask=np.zeros(siz,dtype='int16')
    Vbody=np.zeros(siz,dtype='int16')
    Vlungsmask=np.zeros(siz,dtype='int16')
    Vlung=np.zeros(siz,dtype='int16')
    # Vlungvol=np.zeros((siz[0],1),dtype='int16')
    # Vvesselvol=np.zeros((siz[0],1),dtype='int16')
    Vvesselmask=np.zeros(siz,dtype='int16')
    #Thresholding to create mask. The values are chosen after performing histogram 
    #analysis for all the scans.
    Vmin=np.min(V)
    LungThresh=Vmin+2724
    BodyThresh=Vmin+2524
    VesselThresh=Vmin+2324    
    # i=1
    G1_c=[]
    G2_d=[]
    G3_h=[]
    G4_e=[]
    G5_cr=[]
        # i=0
    for i in range(siz[0]):
        sl=V[i,:,:]#slice wise operation
        Vhline=smooth(sl[:,256],5)
        Vhline1=smooth(sl[256,:],5)
        Vstart=min([Vhline[0],Vhline1[0]])
        V[V==Vmin]=Vstart
        sl=V[i,:,:]
        sl1=np.float32(sl<BodyThresh)
            #kernel definiion at each Morpholigcal operation is mandate for this code
        kernel=np.ones((8,8),np.uint8)
        sl1=cv2.morphologyEx(sl1, cv2.MORPH_CLOSE, kernel)
        # sl1=flood_fill(sl1)
        sl1=imcomplement(sl1)
        sl1=flood_fill(sl1) 
        sl1=sl1.astype('int16')
        Vbodymask[i,:,:]=sl1#Body mask volume
        sl_n_1 = cv2.multiply(sl, sl1)#Masks is applied
        Vbody[i,:,:]=sl_n_1#Body scan volume
        ##Lung Segmentation
        sl2=np.float32(sl_n_1<LungThresh)
        kernel=np.ones((3,3),np.uint8)
        sl2=cv2.erode(sl2,kernel,iterations = 2)
        sl2=flood_fill(sl2)
        sl2=sl2.astype('int16')
        Vlungsmask[i,:,:]=sl2#Lung mask volume
        sl_n_2=cv2.multiply(sl_n_1,sl2)#Lung mask applied
        Vlung[i,:,:]=sl_n_2#Lung scan volume      
        ## Routine for analysis volume and possibly texture
        sl_n_2_t=sl_n_2
        sl_n_2_t = (1*(sl_n_2_t - np.min(sl_n_2_t))/np.ptp(sl_n_2_t)).astype(int)
        # sl_n_2_t = 255 * sl_n_2_t
        Glcmatrix = greycomatrix(sl_n_2_t, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],levels=4)
        G_1_contrast=greycoprops(Glcmatrix, prop='contrast')
        G_2_dissimilarity=greycoprops(Glcmatrix, prop='dissimilarity')
        G_3_homogeneity=greycoprops(Glcmatrix, prop='homogeneity')
        G_4_energy=greycoprops(Glcmatrix, prop='energy')
        G_5_correlation=greycoprops(Glcmatrix, prop='correlation')
        G1_c.append(np.mean(G_1_contrast))
        G2_d.append(np.mean(G_2_dissimilarity))
        G3_h.append(np.mean(G_3_homogeneity))
        G4_e.append(np.mean(G_4_energy))
        G5_cr.append(np.mean(G_5_correlation))
        # Vlungvol[i]=np.count_nonzero(sl2)
        ##Vessel Segmentation
        sl31=sl_n_2!=0
        sl32=sl_n_2>VesselThresh
        sl31=sl31.astype('int16')
        sl32=sl32.astype('int16')
        sl3=cv2.multiply(sl31,sl32)
        sl3=flood_fill(sl3)
        ## Routine for analysis-volume
        # Vvesselvol[i]=np.count_nonzero(sl3)
        Vvesselmask[i,:,:]=sl3
    # Volume calculation
    Vlungvol=np.count_nonzero(Vlungsmask)
    Vvesselvol=np.count_nonzero(Vvesselmask)
    G1_c=sum(G1_c)
    G2_d=sum(G2_d)
    G3_h=sum(G3_h)
    G4_e=sum(G4_e)
    G5_cr=sum(G5_cr)
    ves_lun_rat=(Vvesselvol/Vlungvol)*100
    return Vbody,Vbodymask,Vlung,Vlungsmask,Vvesselmask,Vlungvol,Vvesselvol,G1_c,G2_d,G3_h,G4_e,G5_cr,ves_lun_rat  

#%% Main function
def lung_fun(inputImageFileName,outputImageFileName):
    image,imgA=read_nii_img_file(inputImageFileName)
    Vmin=np.min(imgA)
    Vmax=np.max(imgA)
    Vran=Vmax-Vmin
    hist,bins = np.histogram(imgA.ravel(),Vran,[Vmin,Vmax])
    Vback=bins[np.argmax(hist)]
    if Vback>-2000:
        print('First type function is applied')
        Vbody,Vbodymask,Vlung,Vlungsmask,Vvesselmask,Vlungvol,Vvesselvol,G1_c,G2_d,G3_h,G4_e,G5_cr,ves_lun_rat=lunganalysis_1(imgA)
        sitk.WriteImage(sitk.GetImageFromArray(Vlungsmask), outputImageFileName)
    else:
        print('Second type is applied')  
        # lunganalysis_2(imgA)
        Vbody,Vbodymask,Vlung,Vlungsmask,Vvesselmask,Vlungvol,Vvesselvol,G1_c,G2_d,G3_h,G4_e,G5_cr,ves_lun_rat=lunganalysis_2(imgA)
        sitk.WriteImage(sitk.GetImageFromArray(Vlungsmask), outputImageFileName)
    return Vbody,Vbodymask,Vlung,Vlungsmask,Vvesselmask,Vlungvol,Vvesselvol,G1_c,G2_d,G3_h,G4_e,G5_cr,ves_lun_rat
    

#%% All file analysis
# inputImageFileName='/home/arun/Documents/PyWSPrecision/Brainomix/Images1/vol_02.nii'
# outputImageFileName='vol_02_lungmask.nii'

#Change this to the correct path automatically

# mypath='/home/arun/Documents/PyWSPrecision/Brainomix/Images1/'
fullpath=(os.path.dirname(__file__))
mypath=os.path.join(fullpath, 'Images')

#%%

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# List of features extracted from all 9 images
FeatLis=np.zeros((len(onlyfiles),8))
                 
for file in range(len(onlyfiles)):
# for file in range(1):
# file=0
    filename=onlyfiles[file].split(".")
    inputImageFileName=os.path.join(mypath,onlyfiles[file])
    outputImageFileName=filename[0]+'_mask'+'.nii'
    Vbody,Vbodymask,Vlung,Vlungsmask,Vvesselmask,Vlungvol,Vvesselvol,G1_c,G2_d,G3_h,G4_e,G5_cr,ves_lun_rat=lung_fun(inputImageFileName,outputImageFileName)
    Feat=[Vlungvol,Vvesselvol,ves_lun_rat,G1_c,G2_d,G3_h,G4_e,G5_cr]
    FeatLis[file,:]=Feat


#%% Texture Feature Extraction and Classification of masked lung CT scans
F1=FeatLis[:,4]

F1=(F1-min(F1))/(max(F1)-min(F1))
F2=FeatLis[:,6]

F2=(F2-min(F2))/(max(F2)-min(F2))
F3=FeatLis[:,-1]

F3=(F3-min(F3))/(max(F3)-min(F3))

F12=np.zeros((len(onlyfiles),2))
F12[:,0]=F1
F12[:,1]=F2

F123=np.zeros((len(onlyfiles),3))
F123[:,0]=F1
F123[:,1]=F2
F123[:,2]=F3

#Simple k-means clustering
#2D Feature space
y_pred = KMeans(n_clusters=2, random_state=170).fit_predict(F12)
#3D Feature space
y_pred1 = KMeans(n_clusters=4, random_state=170).fit_predict(F123) 

    

#%% Display/Visualisation 
#Classification

plt.figure(1),
plt.scatter(F1,F2,c=y_pred)
for i in range(len(onlyfiles)):
    plt.annotate(str(i), (F1[i], F2[i]))
plt.xlabel('Normalised Contrast')
plt.ylabel('Normalised Homogeneity')
plt.title('Masked Lung CT - 2D Feature space and Classification')

fig = plt.figure(2)
ax = fig.add_subplot(projection='3d')
for i in range(len(onlyfiles)):
    ax.scatter(F1,F2,F3,c=y_pred1)
    ax.text(F1[i],F2[i],F3[i],  '%s' % (str(i)), size=10, zorder=1,  
    color='k') 
ax.set_xlabel('Normalised Contrast')
ax.set_ylabel('Normalised Homogeneity')
ax.set_zlabel('Normalised Correlation')
ax.set_title('Masked Lung CT - 3D Feature space and Classification')


