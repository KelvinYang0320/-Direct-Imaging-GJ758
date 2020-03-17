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
    # load all images into a image cube
    # and the corresponding parallactic angle in the header
    imgcube = []

    for ind, fn in enumerate(fn_list):
        with fits.open(fn) as img:
            imgcube.append(img[0].data)

    print ("{} files are loaded".format(len(fn_list)))

    return np.array(imgcube)

# path to the directory where the fits files of the raw images are in
route = r"F:\DirectImaging\data-201706\GJ758\GJ758_2.1"
# a list containing all the fits file
fn_list = glob(os.path.join(route, "*.fits"))
# load data here
imgcube = ImgLoader(fn_list)
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
route = r"F:\DirectImaging\data-201706\Calibration\Dark\2000"
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
plt.close()
"""
# Find the centroid.
def Center_of_Gravity(img):
    centroid_x = np.arange(img_size[1]).reshape(1, -1) * img
    centroid_y = np.arange(img_size[0]).reshape(-1, 1) * img
    centroid_x = centroid_x.sum()
    centroid_y = centroid_y.sum()
    a=img.sum()
    return centroid_x / a, centroid_y / a

# Here is an example on what the image will be after registration
foo = np.array(img_calibrated[0, :, :]-np.amin(img_calibrated[0,:,:]))

centroid = Center_of_Gravity(foo)
print ( "The ceneter of original image is:", centroid)

# do shifting here
side=2*(math.ceil(min(img_size[0]/2-abs((img_size[0]/2)-centroid[1]), img_size[1]-abs((img_size[1]/2)-centroid[0]))))
print("The side of the image after registration is",side)
foo_shift=foo[int(centroid[1]-side/2): int(centroid[1]+side/2), int(centroid[0]-side/2):int(centroid[0]+side/2)]
fig = plt.figure(figsize = (12, 8))
plt.ion()
plt.subplot(121)
plt.imshow(foo, cmap= "jet")
plt.scatter(img_size[1]/2,img_size[0]/2 , c = "r", marker = "x")
plt.scatter(centroid[0],centroid[1] , c = "w", marker = "x")
plt.title("Before Registration")
plt.subplot(122)

plt.imshow(foo_shift, cmap = "jet")
plt.title("After Registration")
plt.scatter(side/2, side/2, c = "r", marker = "x")
plt.savefig("Before_After_1") 
plt.show()
plt.pause(2)
plt.close()

# Take the fourier transform of the image
F1=fftpack.fft2(foo_shift)
F1_abs=np.abs(F1)
F2=fftpack.fftshift(F1)
A=np.angle(F2)

plt.ion()
plt.figure(6)
plt.clf()
plt.title(" the original fourier transform image")
plt.imshow(np.log10(np.add(1,np.multiply(1.0,F1_abs))), cmap="jet")
plt.colorbar()
plt.savefig("the original fourier transform image_1")

plt.figure(7)
plt.clf()
plt.title("the phase spectrum of the image")
plt.imshow(A, cmap="jet")
plt.colorbar()
plt.savefig("the phase spectrum of the image_1")
plt.show()

# do shifting here
side=2*50
print("The side of the image after registration is",side)
foo_shift=foo[int(centroid[1]-side/2): int(centroid[1]+side/2), int(centroid[0]-side/2):int(centroid[0]+side/2)]
fig = plt.figure(figsize = (12, 8))
plt.ion()
plt.subplot(121)
plt.imshow(foo, cmap= "jet")
plt.scatter(img_size[1]/2,img_size[0]/2 , c = "r", marker = "x")
plt.scatter(centroid[0],centroid[1] , c = "w", marker = "x")
plt.title("Before Registration")
plt.subplot(122)

plt.imshow(foo_shift, cmap = "jet")
plt.title("After Registration")
plt.scatter(side/2, side/2, c = "r", marker = "x")
plt.savefig("Before_After_2") 
plt.show()
plt.pause(2)
plt.close()

# Take the fourier transform of the image
F1=fftpack.fft2(foo_shift)
F1_abs=np.abs(F1)
F2=fftpack.fftshift(F1)
A=np.angle(F2)

plt.ion()
plt.figure(6)
plt.clf()
plt.title(" the original fourier transform image")
plt.imshow(np.log10(np.add(1,np.multiply(1.0,F1_abs))), cmap="jet")
plt.colorbar()
plt.savefig(" the original fourier transform image_2")

plt.figure(7)
plt.clf()
plt.title("the phase spectrum of the image")
plt.imshow(A, cmap="jet")
plt.colorbar()
plt.savefig("the phase spectrum of the image_2")
plt.show()

# do shifting here
side=2*25
print("The side of the image after registration is",side)
foo_shift=foo[int(227.56-side/2): int(227.56+side/2), int(282.6-side/2):int(282.6+side/2)]
fig = plt.figure(figsize = (12, 8))
plt.ion()
plt.subplot(121)
plt.imshow(foo, cmap= "jet")
plt.scatter(img_size[1]/2,img_size[0]/2 , c = "r", marker = "x")
plt.scatter(282.6,227.56 , c = "w", marker = "x")
plt.title("Before Registration")
plt.subplot(122)

plt.imshow(foo_shift, cmap = "jet")
plt.title("After Registration")
plt.scatter(side/2, side/2, c = "r", marker = "x")
plt.savefig("Before_After_3") 
plt.show()
plt.pause(2)
#plt.close()

# Take the fourier transform of the image
F1=fftpack.fft2(foo_shift)
F1_abs=np.abs(F1)
F2=fftpack.fftshift(F1)
A=np.angle(F2)

plt.ion()
plt.figure(8)
plt.clf()
plt.title(" the original fourier transform image")
plt.imshow(np.log10(np.add(1,np.multiply(1.0,F1_abs))), cmap="jet")
plt.colorbar()
plt.savefig("the original fourier transform image_3")

plt.figure(9)
plt.clf()
plt.title("the phase spectrum of the image")
plt.imshow(A, cmap="jet")
plt.colorbar()
plt.savefig("the phase spectrum of the image_3")
plt.show()
