"""
Calibraiton Process:(no dark flat frames)
Final ReducedObject = (RawObject-MasterDark)/[Flat-MasterFlatDark]normalize
"""
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import fftpack
from glob import glob
from scipy import ndimage
import numpy as np
import math

# load data here
def ImgLoader(fn_list):
    imgcube = []
    DATE=[]
    for ind, fn in enumerate(fn_list):
        with fits.open(fn) as img:
            imgcube.append(img[0].data)
            DATE[ind]=img[0].header["DATE"]
    print ("{} files are loaded".format(len(fn_list)))

    return np.array(imgcube), DATA

# path to the directory where the fits files of the raw images are in
route = r"F:\DirectImaging\data-201706\GJ758\GJ758_1"
# a list containing all the fits file
fn_list = glob(os.path.join(route, "*.fits"))
# load data here
imgcube, DATE = ImgLoader(fn_list)
rawobject=imgcube[:,0,:,:]
print(rawobject.shape)
img_size = rawobject[0, :, :].shape
print(" The size of the img is", img_size)
plt.figure(0)
plt.ion()
plt.clf()
plt.title("the first raw image")
plt.imshow(rawobject[0,:,:], cmap="jet")
plt.colorbar()
plt.savefig("the first raw image")
plt.show() # Show the first raw image.
plt.pause(2)
plt.close()

# path to the directory where the fits files of the Dark are in
route = r"F:\DirectImaging\data-201706\Calibration\Dark\4000"
# a list containing all the fits file
fn_list = glob(os.path.join(route, "*.fits"))
# load data here
imgcube =ImgLoader(fn_list)
dark=imgcube[:,0,:,:]
print(dark.shape)
# master dark
master_dark=np.median(dark, axis=0)
print("The master dark is done.")
plt.figure(1)
plt.ion()
plt.clf()
plt.title("the master dark image")
plt.imshow(master_dark, cmap="jet")
plt.colorbar()
plt.savefig("the master dark image")
plt.show()
plt.pause(2)
plt.close()

# path to the directory where the fits files of the DFlat are in
route = r"F:\DirectImaging\data-201706\Calibration\DFlat"
# a list containing all the fits file
fn_list = glob(os.path.join(route, "*.fits"))
# load data here
imgcube = ImgLoader(fn_list)
DFlat=imgcube[:,0,:,:]

A=np.zeros((5,512,640))
for i in range(len(DFlat[:,0,0])):
    A[i,:,:]=DFlat[i,:,:]/np.median(DFlat[i,:,:])
master_DFlat=np.median(A, axis=0)
master_DFlat=master_DFlat/np.mean(master_DFlat)
print("The master dome flat is done.")
plt.figure(2)
plt.ion()
plt.clf()
plt.title("the master dome flat")
plt.imshow(master_DFlat, cmap="jet")
plt.colorbar()
plt.savefig("the master dome flat")
plt.show()
plt.pause(2)
plt.close()
"""
# Calibrate all images.
img_calibrated=np.zeros((len(rawobject[:,0,0]),512,640))
for i in range(len(rawobject[:,0,0])):
    img_calibrated[i,:,:]=(rawobject[i,:,:]-master_dark)/(master_DFlat)
    print(int((i+1)*100/len(rawobject[:,0,0])),"%")
print("All images have been calibrated.")
plt.figure(3)
plt.ion()
plt.clf()
plt.title("the first calibrated image")
plt.imshow(img_calibrated[0,:,:], cmap="jet")
plt.colorbar()
plt.savefig("the first calibrated image")
plt.show()
plt.pause(2)
plt.close()
"""
# Calibrate all images without dark.
img_calibrated_2=np.zeros((len(rawobject[:,0,0]),512,640))
for i in range(len(rawobject[:,0,0])):
    img_calibrated_2[i,:,:]=(rawobject[i,:,:])/(master_DFlat)
    num=int((i+1)*100/len(rawobject[:,0,0]))
    #if num%10==0:
     #   if num
    #    print(num,"%")
print("All images have been calibrated.")
plt.figure(4)
plt.ion()
plt.clf()
plt.title("the first calibrated image without dark")
plt.imshow(img_calibrated_2[0,:,:], cmap="jet")
plt.colorbar()
plt.savefig("the first calibrated image without dark")
plt.show()
plt.pause(2)
#plt.close()

