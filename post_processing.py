import os
import cv2
import numpy as np
import scipy.ndimage as ndimage
from skimage.transform import hough_circle,hough_circle_peaks
from skimage.draw import circle_perimeter

def hough(edged,limm,limM):
	hough_radii = np.arange(limm, limM, 1)
	hough_res = hough_circle(edged, hough_radii)
	return hough_circle_peaks(hough_res, hough_radii,total_num_peaks=1)


img_path = './.../'
save_path = './.../'

imagelist_1 = os.listdir(img_path)

for lineIdx in range(len(imagelist_1)):

    temp_txt = imagelist_1[lineIdx]
    temp_txt_1 = temp_txt[:-4] + '.bmp'

    if (imagelist_1[lineIdx].endswith(".bmp")):

        bf_img = cv2.imread(img_path + temp_txt,0)
        kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (7, 7))
        print(kernel)

        img = cv2.morphologyEx(bf_img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(save_path + temp_txt,img)

    else:
        continue