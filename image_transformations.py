import numpy as np
import cv2 as cv
import math
import random

#Brightness control values
alpha_vals = [3.5, 2/7]
#Original photos counter
i = 1
#New photos counter
ni = 1

image = cv.imread('/Users/michber/Desktop/Studia/Praca_inzynierska/Projekt/originals/1_train/{:04d}.png'.format(i))

#Fingers photos - bottom-left, upper-right
finger_bl = cv.imread('/Users/michber/Desktop/Studia/Praca_inzynierska/Projekt/Palce/palec_ld.png')
finger_ur = cv.imread('/Users/michber/Desktop/Studia/Praca_inzynierska/Projekt/Palce/palec_pg.png')

while image is not None:
    #Resize images
    width = int(image.shape[1]/4)
    height = int(image.shape[0]/4)
    dims = (width, height)
    image = cv.resize(image, dims, interpolation = cv.INTER_AREA)

    #List of empty canvases for edited photos
    new_images = [np.zeros(image.shape, image.dtype) for p in range(7)]

    #Photo editors
    #Lightening and darkening
    for alpha, b in zip(alpha_vals, range(2)):
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_images[b][y,x,c] = np.clip(alpha*image[y,x,c], 0, 255)

    #Bluring with Gaussian
    new_images[2] = cv.GaussianBlur(image,(9,9),0)

    #Adding noise
    #Gaussian noise
    # row,col,ch= image.shape
    # mean = 0
    # var = 256
    # sigma = var**0.5
    # gauss = np.random.normal(mean,sigma,(row,col,ch))
    # gauss = gauss.reshape(row,col,ch)
    # new_images[7] = image + gauss

    #Salt&Pepper noise
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
    out[coords] = 1
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image.shape]
    out[coords] = 0
    new_images[3] = out

    # #Poisson noise
    # new_images[9] = np.random.poisson(image)
    #
    # #Speckle noise
    # row,col,ch = image.shape
    # gauss = np.random.randn(row,col,ch)
    # gauss = gauss.reshape(row,col,ch)
    # new_images[10] = image + image * gauss

    #Finger in the frame
    #Bottom-left
    diff_y = max(image.shape[0] - finger_bl.shape[0],0)

    for y in range(finger_bl.shape[0]):
        for x in range(finger_bl.shape[1]):
            if y >= image.shape[0] or x >= image.shape[1]:
                continue
            else:
                for c in range(image.shape[2]):
                    if finger_bl[y,x,0] >= 5 and finger_bl[y,x,0] <= 55 and finger_bl[y,x,1] >= 15 and finger_bl[y,x,0] <= 75 and finger_bl[y,x,2] >= 60 and finger_bl[y,x,2] <= 115:
                        new_images[4][y+diff_y,x,c] = finger_bl[y,x,c]
                    else:
                        new_images[4][y,x,c] = image[y,x,c]


    for y in range(new_images[4].shape[0]):
        for x in range(new_images[4].shape[1]):
                if new_images[4][y,x,0] == 0 and new_images[4][y,x,1] == 0 and new_images[4][y,x,2] == 0:
                        new_images[4][y,x,:] = image[y,x,:]

    #Upper-right
    diff_x = max(image.shape[1] - finger_ur.shape[1],0)

    for y in range(int(finger_ur.shape[0])):
        for x in range(int(finger_ur.shape[1])):
            for c in range(image.shape[2]):
                if y >= image.shape[0] or x >= image.shape[1]:
                    continue
                else:
                    if finger_ur[y,x,0] >= 20 and finger_ur[y,x,0] <= 80 and finger_ur[y,x,1] >= 35 and finger_ur[y,x,0] <= 90 and finger_ur[y,x,2] >= 70 and finger_ur[y,x,2] <= 120:
                        new_images[5][y,x+diff_x,c] = finger_ur[y,x,c]
                    else:
                        new_images[5][y,x,c] = image[y,x,c]

    for y in range(new_images[5].shape[0]):
        for x in range(new_images[5].shape[1]):
            if new_images[5][y,x,0] == 0 and new_images[5][y,x,1] == 0 and new_images[5][y,x,2] == 0:
                new_images[5][y,x,:] = image[y,x,:]

    #Segmentation
    # new_images[13] = cv.pyrMeanShiftFiltering(image, 20, 45, 3)

    #Reducing contrast
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_images[6][y,x,c] = ((image[y,x,c]*(110-70))/110)+70

    cv.imwrite('/Users/michber/Desktop/Studia/Praca_inzynierska/Projekt/dataset/1_train/good/{:04d}.png'.format(ni), image)
    ni=ni+1
    for nim in new_images:
        cv.imwrite('/Users/michber/Desktop/Studia/Praca_inzynierska/Projekt/dataset/1_train/bad/{:04d}.png'.format(ni), nim)
        ni=ni+1

    i+=1
    image = cv.imread('/Users/michber/Desktop/Studia/Praca_inzynierska/Projekt/originals/1_train/{:04d}.png'.format(i))
