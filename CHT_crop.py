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
#train image
train_bf_path = './data/crop400/train_first/' ### Coarse segmentation result path
train_path = './data/crop400/REFUGE-Training400/' ### Original Image
label_path = './data/crop400/Annotation-Training400/Annotation-Training400/Disc_Cup_Masks/' ##Original Image label
#val img
val_bf_path = './.../'
val_path = './.../'
val_label_path ='./.../'
#test imag
test_bf_path = './.../'
test_path = './.../'
test_label_path = './.../'
####save path
train_save_path = './.../'
label_save_path = './.../'

val_save_path = './.../'
val_label_save_path = './.../'

test_save_path = './.../'
test_label_save_path = './.../'


imagelist_1 = os.listdir(train_path)
imagelist_2 = os.listdir(val_path)
imagelist_3 = os.listdir(test_path)


# crop train data Glaucoma

# #---train-----
for lineIdx in range(len(imagelist_1)):

    temp_txt = imagelist_1[lineIdx]
    print(temp_txt)
    temp_txt_1 = temp_txt[:-4] + '.bmp'

    if (imagelist_1[lineIdx].endswith(".jpg")):
        bf_img = cv2.imread(train_bf_path + temp_txt)
        org_img = cv2.imread(train_path + temp_txt)
        org_label = cv2.imread(label_path + temp_txt_1,0)
        gray_r = bf_img
        gray_blur = cv2.GaussianBlur(gray_r, (5, 5), 0)
        edged = cv2.Canny(gray_blur, 15, 35)
        # cv2.imshow('edged1',edged)
        kernel = np.ones((3, 3), np.uint8)
        edged = cv2.dilate(edged, kernel)
        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, 1500, param1=100, param2=10, minRadius=60,
                                   maxRadius=200)
        print('The centre point of img ' + str(temp_txt) + ' is ' + str(circles))
        circles = circles[0, 0]
        circles = np.uint16(np.around(circles))
        print(circles)

        x0 = circles[0] - 200
        if x0 < 0:
            x0 = 0
        y0 = circles[1] - 200
        if y0 < 0:
            y0 = 0
        x1 = circles[0] + 200
        if x1 > org_img.shape[0]:
            x1 = org_img.shape[0]
        y1 = circles[1] + 200
        if y1 > org_img.shape[1]:
            y1 = org_img.shape[1]


        #裁剪图片
        crop_img = org_img[y0:y1, x0:x1]
        crop_label = org_label[y0:y1,x0:x1]
        img_shape = crop_img.shape
        x_shape = img_shape[0]
        y_shape = img_shape[1]
        if x_shape <= 400:
            pad_xnum = 400 - x_shape
        if y_shape <= 400:
            pad_ynum = 400 - y_shape
        if pad_xnum > 0 or pad_ynum > 0:
            crop_img = cv2.copyMakeBorder(crop_img,pad_xnum,0,pad_ynum,0,cv2.BORDER_CONSTANT,value=0)
            crop_label = cv2.copyMakeBorder(crop_label,pad_xnum,0,pad_ynum,0,cv2.BORDER_CONSTANT,value=255)

        # cv2.imshow('crop_label',crop_label)
        cv2.imwrite(train_save_path + temp_txt,crop_img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
        cv2.imwrite(label_save_path + temp_txt_1,crop_label)
        # cv2.imwrite(label_save_path + temp_txt, crop_label)
        print(crop_img.shape)
        print((crop_label.shape))

    else:
        continue
#------val----------
# for lineIdx in range(len(imagelist_2)):
#
#     temp_txt = imagelist_2[lineIdx]
#     print(temp_txt)
#     temp_txt_1 = temp_txt[:-4] + '.bmp'
#
#     if (imagelist_2[lineIdx].endswith(".jpg")):
#         bf_img = cv2.imread(val_bf_path + temp_txt)
#         org_img = cv2.imread(val_path + temp_txt)
#         org_label = cv2.imread(val_label_path + temp_txt_1,0)
#
#         gray_r = bf_img
#         gray_blur = cv2.GaussianBlur(gray_r, (5, 5), 0)
#
#         edged = cv2.Canny(gray_blur, 15, 35)
#         kernel = np.ones((3, 3), np.uint8)
#         edged = cv2.dilate(edged, kernel)
#
#         circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, 2000, param1=100, param2=11, minRadius=40,
#                                    maxRadius=150)
#         print('The centre point of img ' + str(temp_txt) + ' is ' + str(circles))
#         circles = circles[0, 0]
#         circles = np.uint16(np.around(circles))
#         print(circles)
#
#         x0 = circles[0] - 200
#         if x0 < 0:
#             x0 = 0
#         y0 = circles[1] - 200
#         if y0 < 0:
#             y0 = 0
#         x1 = circles[0] + 200
#         if x1 > org_img.shape[0]:
#             x1 = org_img.shape[0]
#         y1 = circles[1] + 200
#         if y1 > org_img.shape[1]:
#             y1 = org_img.shape[1]
#
#         #裁剪图片
#         crop_img = org_img[y0:y1, x0:x1]
#         crop_label = org_label[y0:y1,x0:x1]
#         img_shape = crop_img.shape
#         x_shape = img_shape[0]
#         y_shape = img_shape[1]
#         if x_shape <= 400:
#             pad_xnum = 400 - x_shape
#         if y_shape <= 400:
#             pad_ynum = 400 - y_shape
#         #若裁剪的图片尺寸不是480×480 则在边缘补值
#         if pad_xnum > 0 or pad_ynum > 0:
#             crop_img = cv2.copyMakeBorder(crop_img,pad_xnum,0,pad_ynum,0,cv2.BORDER_CONSTANT,value=0)
#             crop_label = cv2.copyMakeBorder(crop_label,pad_xnum,0,pad_ynum,0,cv2.BORDER_CONSTANT,value=255)
#
#         # cv2.imshow('crop_label',crop_label)
#         cv2.imwrite(val_save_path + temp_txt,crop_img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#         cv2.imwrite(val_label_save_path + temp_txt_1,crop_label)
#         # cv2.imwrite(label_save_path + temp_txt, crop_label)
#         print(crop_img.shape)
#         print((crop_label.shape))
#
#     else:
#         continue


#----test-----
'''
cricles_list_test = {}
center ={}
for lineIdx in range(len(imagelist_3)):

    temp_txt = imagelist_3[lineIdx]
    print(temp_txt)
    temp_txt_1 = temp_txt[:-4] + '.bmp'
    if (imagelist_3[lineIdx].endswith(".jpg")):
        bf_img = cv2.imread(test_bf_path + temp_txt)
        org_img = cv2.imread(test_path + temp_txt)
        org_label = cv2.imread(test_label_path + temp_txt_1,0)

        gray_r = bf_img
        gray_blur = cv2.GaussianBlur(gray_r, (5, 5), 0)

        edged = cv2.Canny(gray_blur, 15, 35)
        kernel = np.ones((3, 3), np.uint8)
        edged = cv2.dilate(edged, kernel)

        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 10, 1500, param1=100, param2=10, minRadius=60,
                                   maxRadius=200)
        print('The centre point of img ' + str(temp_txt) + ' is ' + str(circles))
        circles = circles[0, 0]
        cricles_list_test[temp_txt_1] = circles
        center = []

        circles = np.uint16(np.around(circles))
        print(circles)

        x0 = circles[0] - 200
        if x0 < 0:
            x0 = 0
        y0 = circles[1] - 200
        if y0 < 0:
            y0 = 0
        x1 = circles[0] + 200
        if x1 > org_img.shape[0]:
            x1 = org_img.shape[0]
        y1 = circles[1] + 200
        if y1 > org_img.shape[1]:
            y1 = org_img.shape[1]

        #裁剪图片
        crop_img = org_img[y0:y1, x0:x1]
        crop_label = org_label[y0:y1,x0:x1]
        img_shape = crop_img.shape
        x_shape = img_shape[0]
        y_shape = img_shape[1]
        if x_shape <= 400:
            pad_xnum = 400 - x_shape
        if y_shape <= 400:
            pad_ynum = 400 - y_shape
        #若裁剪的图片尺寸不是480×480 则在边缘补值
        if pad_xnum > 0 or pad_ynum > 0:
            crop_img = cv2.copyMakeBorder(crop_img,pad_xnum,0,pad_ynum,0,cv2.BORDER_CONSTANT,value=0)
            crop_label = cv2.copyMakeBorder(crop_label,pad_xnum,0,pad_ynum,0,cv2.BORDER_CONSTANT,value=255)

        # cv2.imshow('crop_label',crop_label)
        cv2.imwrite(test_save_path + temp_txt,crop_img,[int(cv2.IMWRITE_JPEG_QUALITY),100])
        cv2.imwrite(test_label_save_path + temp_txt_1,crop_label)
        # cv2.imwrite(label_save_path + temp_txt, crop_label)
        print(crop_img.shape)
        print((crop_label.shape))

    else:
        continue
s1 = str(cricles_list_test)
f = open('circles_list_test.txt','a')
f.writelines(s1)
f.write('\r\n')
f.close()

# crop test data Non-Glaucoma
'''
'''
for lineIdx in range(len(imagelist_2)):

    temp_txt = imagelist_2[lineIdx]
    temp_txt_1 = temp_txt[:-4] + '.bmp'

    if (imagelist_2[lineIdx].endswith(".jpg")):

        org_img = cv2.imread(train_path_2 + temp_txt)
        # cv2.imshow('org_img',org_img)
        org_label = cv2.imread(label_path_2 + temp_txt_1)
        # cv2.imshow('org_label',org_label)
        b, g, r = cv2.split(org_img)
        gray_r = r
        gray_blur = cv2.GaussianBlur(gray_r, (5, 5), 0)
        gray_r = cv2.addWeighted(gray_r, 1.5, gray_blur, -0.5, 0, gray_r)
        # cv2.imshow('gray_r',gray_r)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        # print(kernel)
        # gray = ndimage.grey_closing(gray_r,structure=kernel)
        gray = cv2.morphologyEx(gray_r, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('gray',gray)
        # gray = cv2.equalizeHist(gray)
        # cv2.imshow('gray',gray)
        edged = cv2.Canny(gray, 9, 35)
        # cv2.imshow('edged1',edged)
        kernel = np.ones((2, 2), np.uint8)
        edged = cv2.dilate(edged, kernel)
        # print(edged.dtype)
        # edged = cv2.GaussianBlur(edged, (5,5), 0)
        # edged = np.uint8(edged)
        # accums,cx,cy,radii = hough(edged,55,80)
        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, 1500, param1=10, param2=52, minRadius=60,
                                   maxRadius=178)
        # cv2.imshow('edged2',edged)
        # print(circles)

        circles = np.uint16(np.around(circles))
        circles = circles[0, 0]

        x0 = circles[0] - 300
        if x0 < 0:
            x0 = 0
        y0 = circles[1] - 300
        if y0 < 0:
            y0 = 0
        x1 = circles[0] + 300
        if x1 > org_img.shape[0]:
            x1 = org_img.shape[0]
        y1 = circles[1] + 300
        if y1 > org_img.shape[1]:
            y1 = org_img.shape[1]

        crop_img = org_img[y0:y1, x0:x1]
        crop_label = org_label[y0:y1,x0:x1]
        # cv2.imshow('crop_label',crop_label)
        cv2.imwrite(train_save_path + temp_txt,crop_img)
        cv2.imwrite(label_save_path + temp_txt_1,crop_label)
        # cv2.imwrite(label_save_path + temp_txt, crop_label)

    else:
        continue
'''

'''
# crop val data
for lineIdx in range(len(imagelist_3)):

    temp_txt = imagelist_3[lineIdx]
    temp_txt_1 = temp_txt[:-4] + '.bmp'

    if (imagelist_3[lineIdx].endswith(".jpg")):

        org_img = cv2.imread(val_path + temp_txt)
        # cv2.imshow('org_img',org_img)
        org_label = cv2.imread(val_label_path + temp_txt_1)
        # cv2.imshow('org_label',org_label)
        b, g, r = cv2.split(org_img)
        gray_r = r
        gray_blur = cv2.GaussianBlur(gray_r, (5, 5), 0)
        gray_r = cv2.addWeighted(gray_r, 1.5, gray_blur, -0.5, 0, gray_r)
        # cv2.imshow('gray_r',gray_r)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        # print(kernel)
        # gray = ndimage.grey_closing(gray_r,structure=kernel)
        gray = cv2.morphologyEx(gray_r, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow('gray',gray)
        # gray = cv2.equalizeHist(gray)
        # cv2.imshow('gray',gray)
        edged = cv2.Canny(gray, 9, 35)
        # cv2.imshow('edged1',edged)
        kernel = np.ones((2, 2), np.uint8)
        edged = cv2.dilate(edged, kernel)
        # print(edged.dtype)
        # edged = cv2.GaussianBlur(edged, (5,5), 0)
        # edged = np.uint8(edged)
        # accums,cx,cy,radii = hough(edged,55,80)
        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, 1500, param1=10, param2=52, minRadius=60,
                                   maxRadius=178)
        # cv2.imshow('edged2',edged)
        # print(circles)

        circles = np.uint16(np.around(circles))
        circles = circles[0, 0]

        x0 = circles[0] - 300
        if x0 < 0:
            x0 = 0
        y0 = circles[1] - 300
        if y0 < 0:
            y0 = 0
        x1 = circles[0] + 300
        if x1 > org_img.shape[0]:
            x1 = org_img.shape[0]
        y1 = circles[1] + 300
        if y1 > org_img.shape[1]:
            y1 = org_img.shape[1]

        crop_img = org_img[y0:y1, x0:x1]
        crop_label = org_label[y0:y1, x0:x1]
        # cv2.imshow('crop_label',crop_label)
        cv2.imwrite(val_save_path + temp_txt, crop_img)
        cv2.imwrite(val_label_save_path + temp_txt_1, crop_label)
        # cv2.imwrite(label_save_path + temp_txt, crop_label)

    else:
        continue

cv2.destroyAllWindows()

#debug single image
'''
'''
temp_txt = imagelist_1[1]
# load image
temp_txt_1 = temp_txt[:-4]+'.bmp'
#
org_img = cv2.imread(train_path_1 + temp_txt)
org_label = cv2.imread(label_path_1 + temp_txt_1)
# cv2.imshow('org_label',org_label)
# cv2.imshow('org_label',org_label)
b, g, r = cv2.split(org_img)
gray_r = r
gray_blur = cv2.GaussianBlur(gray_r, (5, 5), 0)
gray_r = cv2.addWeighted(gray_r, 1.5, gray_blur, -0.5, 0, gray_r)
# cv2.imshow('gray_r',gray_r)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
# print(kernel)
# gray = ndimage.grey_closing(gray_r,structure=kernel)
gray = cv2.morphologyEx(gray_r, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('gray',gray)
# gray = cv2.equalizeHist(gray)
# cv2.imshow('gray',gray)
edged = cv2.Canny(gray, 15, 35)
# cv2.imshow('edged1',edged)
kernel = np.ones((3, 3), np.uint8)
edged = cv2.dilate(edged, kernel)
# print(edged.dtype)
# edged = cv2.GaussianBlur(edged, (5,5), 0)
# edged = np.uint8(edged)
# accums,cx,cy,radii = hough(edged,55,80)
circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, 1500, param1=100, param2=25, minRadius=80,
                           maxRadius=180)
# cv2.imshow('edged2',edged)
# print(circles)
circles = np.uint16(np.around(circles))
circles = circles[0, 0]

x0 = circles[0] - 300
if x0 < 0:
    x0 = 0
y0 = circles[1] - 300
if y0 < 0:
    y0 = 0
x1 = circles[0] + 300
if x1 > org_img.shape[0]:
    x1 = org_img.shape[0]
y1 = circles[1] + 300
if y1 > org_img.shape[1]:
    y1 = org_img.shape[1]

# crop_img = org_img[y0:y1, x0:x1]
crop_label = org_label[y0:y1,x0:x1]
cv2.imshow('crop_label',crop_label)
cv2.imwrite('crop_label.bmp',crop_label)
# cv2.imwrite(train_save_path + temp_txt,crop_img)
# cv2.imwrite(label_save_path + temp_txt_1,crop_label)
cv2.waitKey(0)
'''




