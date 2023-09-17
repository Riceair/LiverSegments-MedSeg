import matplotlib.pyplot as plt
import numpy as np
import cv2

def apply_ct_window(img, window):
    # window = (window width, window level)
    R = (img-window[1]+0.5*window[0])/window[0]
    R[R<0] = 0
    R[R>1] = 1
    return R

def getCTRGB(img):
    img_rgb = apply_ct_window(img, [400,50])
    img_rgb = (img_rgb*255).astype('uint8')
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    return img_rgb

def getMaskRGB(mask):
    MASK_COLOR = np.array([[0,75,135], [255,205,0], [0,106,68], [193,39,45], [0,173,190] 
                    ,[166,0,176], [204,104,0], [62,95,28], [94,87,153]])
    display_img = np.zeros(mask.shape + (3,))

    for i in range(len(MASK_COLOR)):
        indices = np.where(mask == i+1)
        display_img[indices] = MASK_COLOR[i]
    return display_img

def getCTwithMaskLiverTumor(img_rgb, mask):
    MASK_COLOR = np.array([[0, 0, 255], [255, 0, 0], [0, 255, 0]])
    display_img = img_rgb.copy()

    for i in range(len(MASK_COLOR)):
        indices = np.where(mask == i+1)
        display_img[indices] = MASK_COLOR[i]
    return display_img

def getCTwithMask(img_rgb, mask): #input rgb img, mask(with 0-8 labels)
    MASK_COLOR = np.array([[0,75,135], [255,205,0], [0,106,68], [193,39,45], [0,173,190] 
                    ,[166,0,176], [204,104,0], [62,95,28], [94,87,153]])
    display_img = img_rgb.copy()

    for i in range(len(MASK_COLOR)):
        indices = np.where(mask == i+1)
        display_img[indices] = MASK_COLOR[i]
    return display_img

def pltNImages(imgs, cmap=None): #input image list
    img_num = len(imgs)
    fig = plt.figure()
    for i in range(img_num):
        ax = fig.add_subplot(1, img_num, i+1)
        if cmap == None:
            ax.imshow(imgs[i])
            plt.axis("off")
        else:
            ax.imshow(imgs[i], cmap=cmap)
            plt.axis("off")
    fig.tight_layout()
    plt.show()

def pltSingleImage(img, cmap=None):
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()

def saveNImages(imgs, save_path, cmap=None): #input image list
    img_num = len(imgs)
    fig = plt.figure()
    for i in range(img_num):
        ax = fig.add_subplot(1, img_num, i+1)
        if cmap == None:
            ax.imshow(imgs[i])
        else:
            ax.imshow(imgs[i], cmap=cmap)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)