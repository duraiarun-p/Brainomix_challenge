#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:50:24 2022

@author: arun
"""
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import sys, getopt
from operator import sub

# import gui
#%% Different set of funcitons
def normalise_image(image_sitk):
    """
    :param image_sitk:
    :return:
    """
    # suppress an pixel less than 20-percentile to be a background and vice versa
    image_array = sitk.GetArrayFromImage(image_sitk)
    pixels = image_array.ravel()
    q20 = np.quantile(pixels, 0.2)
    q90 = np.quantile(pixels, 0.9)
    norm_image = sitk.Clamp(image_sitk, lowerBound=q20, upperBound=q90)
    norm_image = (norm_image - pixels.mean()) / pixels.std()
    # return sitk.RescaleIntensity(norm_image)
    return norm_image


###############################################################################
def segment_body(image_sitk):
    """
    :param image_sitk:
    :return:
    """
    # select seed point in the background
    seed = image_sitk.GetSize()
    seed = tuple(map(sub, seed, (1, 1, 1)))
    # region growing from the seed point
    seg_con = sitk.ConnectedThreshold(image_sitk, seedList=[seed], lower=75, upper=100)
    # sitk.WriteImage(seg_con, 'seg_con.nii.gz')
    # some morphological operations to get rid of isolated islands in the background
    vectorRadius = (10, 10, 10)
    kernel = sitk.sitkBall
    seg_clean = sitk.BinaryMorphologicalClosing(seg_con, vectorRadius, kernel)
    # sitk.WriteImage(seg_clean, 'seg_clean.nii.gz')
    # reverse background mask values to get the body mask
    body_mask_0 = seg_clean == 0
    # more morphological operations to clean the body mask
    vectorRadius = (3, 3, 3)
    body_mask_0 = sitk.BinaryMorphologicalOpening(body_mask_0, vectorRadius, kernel)
    # sitk.WriteImage(body_mask_0, 'body_mask_0.nii.gz')
    print('Refining body mask...')
    # find biggest connected component, which is supposed to be the body
    body_mask = sitk.ConnectedComponent(body_mask_0)
    # sitk.WriteImage(body_mask, 'body_mask_1.nii.gz')
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(body_mask)
    # filter out smaller components
    label_sizes = [stats.GetNumberOfPixels(l) for l in stats.GetLabels()]
    biggest_labels = np.argsort(label_sizes)[::-1]
    return body_mask == stats.GetLabels()[biggest_labels[0]]  # biggest component has the highest label value


###############################################################################
def segment_lungs(image_stik):
    """
    :param image_stik:
    :return:
    """
    # Binary threshold
    extracted_lungs_0 = sitk.BinaryThreshold(image_stik, lowerThreshold=20., upperThreshold=50.)
    # sitk.WriteImage(extracted_lungs_0, 'extracted_lungs_0.nii.gz')
    # some morphological operations to get rid of isolated islands in the background
    vectorRadius = (20, 20, 20)
    kernel = sitk.sitkBall
    extracted_lungs_1 = sitk.BinaryMorphologicalClosing(extracted_lungs_0, vectorRadius, kernel)
    vectorRadius = (2, 2, 2)
    extracted_lungs_1 = sitk.BinaryMorphologicalOpening(extracted_lungs_1, vectorRadius, kernel)
    # sitk.WriteImage(extracted_lungs_1, 'extracted_lungs_1.nii.gz')
    # find biggest connected component, which is supposed to be the body
    extracted_lungs_2 = sitk.ConnectedComponent(extracted_lungs_1)
    # sitk.WriteImage(extracted_lungs_2, 'extracted_lungs_2.nii.gz')
    # find biggest components
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(extracted_lungs_2)
    # filter out smaller components
    label_sizes = [stats.GetNumberOfPixels(l) for l in stats.GetLabels()]
    biggest_labels = np.argsort(label_sizes)[::-1]
    # biggest two components are the right and left lungs
    right_lung = extracted_lungs_2 == stats.GetLabels()[biggest_labels[0]]
    left_lung = extracted_lungs_2 == stats.GetLabels()[biggest_labels[1]]
    # some morphological operations to get rid of isolated islands in the background
    print('Refining lung masks...')
    left_lung = sitk.BinaryFillhole(left_lung)
    right_lung = sitk.BinaryFillhole(right_lung)
    vectorRadius = (20, 20, 20)
    right_lung = sitk.BinaryMorphologicalClosing(right_lung, vectorRadius, kernel)
    left_lung = sitk.BinaryMorphologicalClosing(left_lung, vectorRadius, kernel)
    vectorRadius = (2, 2, 2)
    right_lung = sitk.BinaryMorphologicalOpening(right_lung, vectorRadius, kernel)
    left_lung = sitk.BinaryMorphologicalOpening(left_lung, vectorRadius, kernel)
    vectorRadius = (20, 20, 20)
    right_lung = sitk.BinaryMorphologicalClosing(right_lung, vectorRadius, kernel)
    left_lung = sitk.BinaryMorphologicalClosing(left_lung, vectorRadius, kernel)
    # dilate the mask 2 pixels to recover the smoothing effect
    right_lung = sitk.BinaryDilate(right_lung, 2, kernel)
    left_lung = sitk.BinaryDilate(left_lung, 2, kernel)
    return right_lung + 2 * left_lung  # return merged labels
#%%
def read_nii_img_file(inputImageFileName):
    image=sitk.ReadImage(inputImageFileName)
    imgA=sitk.GetArrayFromImage(image)
    return image,imgA




#%% Functions
class lungseg(): 
    
    def __init__(self,img):
        self.img = img
        self.temp_img = None
        self.img_uint8 = None
    
    def conv_2_uint8(self, WINDOW_LEVEL=(500,1050)):
        """
        Convert original image to 8-bit image
        :param WINDOW_LEVEL: Using an external viewer (ITK-SNAP or 3DSlicer)
                             we identified a visually appealing window-level setting
        :return: None
        """
        # self.img_uint8 = sitk.Cast(self.img,
        #                           sitk.sitkUInt8)
        self.img_uint8 = sitk.Cast(sitk.IntensityWindowing(self.img,
                                  windowMinimum=WINDOW_LEVEL[1] - WINDOW_LEVEL[0] / 2.0,
                                  windowMaximum=WINDOW_LEVEL[1] + WINDOW_LEVEL[0] / 2.0),
                                  sitk.sitkUInt8)
    
    def regiongrowing(self, seed_pts):
        """
        Implement ConfidenceConnected by SimpleITK tools with given seed points
        :param seed_pts: seed points for region growing [(z,y,x), ...]
        :return: None
        """
        self.temp_img = sitk.ConfidenceConnected(self.img, seedList=seed_pts,
                                                           numberOfIterations=0,
                                                           multiplier=2,
                                                           initialNeighborhoodRadius=1,
                                                           replaceValue=1)

    # def image_showing(self, title=''):
    #     """
    #     Showing image.
    #     :return: None
    #     """
    #     gui.MultiImageDisplay(image_list=[sitk.LabelOverlay(self.img_uint8, self.temp_img)],
    #                           title_list=[title])

    def image_closing(self, size=7):
        """
        Implement morphological closing to fix the "holes" inside the image.
        :param size: the size the closing kernel
        :return: None
        """
        closing = sitk.BinaryMorphologicalClosingImageFilter()
        closing.SetForegroundValue(1)
        closing.SetKernelRadius(size)
        self.temp_img = closing.Execute(self.temp_img)


#%% Data loading

inputImageFileName='/home/arun/Documents/PyWSPrecision/Brainomix/Images1/vol_01.nii'

image,imgA=read_nii_img_file(inputImageFileName)

# image = sitk.ReadImage(inputImageFileName)
# ls=lungseg(image)
# image_1=ls.conv_2_uint8(image)

sl_h=imgA[0,256,:]
sl_v=imgA[0,:,256]

norm_image_sitk = normalise_image(image)
# smooth_image_sitk = sitk.SmoothingRecursiveGaussian(image, 2.)
smooth_image_sitk = sitk.SmoothingRecursiveGaussian(norm_image_sitk, 2.)
body_sitk = segment_body(smooth_image_sitk)

body_sitk_arr=sitk.GetArrayFromImage(body_sitk)
#%% Sample visualisation



norm_image_sitk_arr=sitk.GetArrayFromImage(norm_image_sitk)
img_sl_1=norm_image_sitk_arr[10,:,:]
body_arr_sl=body_sitk_arr[10,:,:]
plt.figure(1),
plt.subplot(1,2,1)
plt.imshow(img_sl_1, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(body_arr_sl)

sl_h=norm_image_sitk_arr[0,256,:]
sl_v=norm_image_sitk_arr[0,:,256]

plt.figure(2),
plt.subplot(121),
plt.plot(sl_h)
plt.subplot(122),
plt.plot(sl_v)
#%%
#%% Segmentation
# index=0
# plt.figure(3),
# plt.subplot(2,3,1),
# plt.imshow(Vbodymask[index,:,:],cmap='gray')
# plt.title('(a) Body Mask')
# plt.subplot(2,3,2),
# plt.imshow(Vbody[index,:,:],cmap='gray')
# plt.title('(b) Masked body')
# plt.subplot(2,3,3),
# plt.imshow(Vlungsmask[index,:,:],cmap='gray')
# plt.title('(c) Lung mask')
# plt.subplot(2,3,4),
# plt.imshow(Vlung[index,:,:],cmap='gray')
# plt.title('(d) Masked lungs')
# plt.subplot(2,3,5),
# plt.imshow(Vvesselmask[index,:,:],cmap='gray')
# plt.title('(e) Vessel mask')