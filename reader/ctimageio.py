import matplotlib.pyplot as plt
import SimpleITK as sitk
import skimage.io as io
import numpy as np
import pydicom

def readImage(path):
    img = sitk.ReadImage(path)
    #轉為np array
    data = sitk.GetArrayFromImage(img)
    return data

def getSliceLocation(path):
    img = pydicom.dcmread(path)
    return img.SliceLocation

def showImage(data, index):
    io.imshow(data[index], cmap='gray')
    io.show()

def show2Images(img1, img2):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(img1, cmap='gray')
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(img2, cmap='gray')
    plt.show()