import numpy as np
import cv2
import os



def return_list(data_path,data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    return file_list

def contrast_img(img,c,b): #定义亮度调节函数
    blank = np.zeros([rows,cols,channels],img.dtype)
    dst = cv2.addWeighted(img,c,blank,1-c,b)
    return dst

def SaltAndPepper(src,percetage):   #定义添加椒盐噪声的函数
    SP_NoiseImg=src
    SP_NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(SP_NoiseNum):
        randX=random.random_integers(0,src.shape[0]-1)
        randY=random.random_integers(0,src.shape[1]-1)
        if random.random_integers(0,1)==0:
            SP_NoiseImg[randX,randY]=0
        else:
            SP_NoiseImg[randX,randY]=255
    return SP_NoiseImg

def addGaussianNoise(image,percetage):  #定义添加高斯噪声的函数
    G_Noiseimg = image
    G_NoiseNum=int(percetage*image.shape[0]*image.shape[1])
    for i in range(G_NoiseNum):
        temp_x = np.random.randint(0,image.shape[0])
        temp_y = np.random.randint(0,image.shape[1])
        G_Noiseimg[temp_x][temp_y] = 255
    return G_Noiseimg


data_path = './data/crop'
train_img_path = data_path + '/img/img800/img_M/'
label_img_path = data_path + '/img/img800/label_M/'
weight_img_path = data_path +'/img/img800/weight_M/'
save_path_img = data_path +'/img/img800/img_F/'
save_path_label = data_path +'/img/img800/label_F/'
save_path_weight = data_path + '/img/img800/weight_F/'
file_list = return_list(train_img_path, '.jpg')
n = len(file_list)
print(n)
#######################################################################
'''加噪声'''
for i in range(n):
    temp_list = file_list[i]
    img_name = os.path.join(train_img_path,temp_list[:-4]+'.jpg')
    label_name = os.path.join(label_img_path,temp_list[:-4]+'.bmp')
    weight_name = os.path.join(weight_img_path,temp_list[:-4]+'.bmp')

    img = cv2.imread(img_name)
    label = cv2.imread(label_name,0)
    weight = cv2.imread(weight_name,0)


    # rows,cols,channels = img.shape
    # im = np.empty((rows,cols),np.uint8)
    r, g, b = cv2.split(img)
    r1 = addGaussianNoise(r, 0.005)
    g1 = addGaussianNoise(g, 0.005)
    b1 = addGaussianNoise(b, 0.005)  # 添加0.5%的高斯噪声
    img_noise = cv2.merge([r1,g1,b1])

    cv2.imwrite(save_path_img + 'N_{}.jpg'.format(i), img_noise, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(save_path_label + 'N_{}.bmp'.format(i), label)
    cv2.imwrite(save_path_weight + 'N_{}.bmp'.format(i), label)

    print(i+1)

##############################################################
'''亮度调节'''
# for i in range(n):
#
#     temp_list = file_list[i]

#     img_name = os.path.join(train_img_path,temp_list[:-4]+'.jpg')
#     label_name = os.path.join(label_img_path,temp_list[:-4]+'.bmp')
#     weight_name = os.path.join(weight_img_path,temp_list[:-4]+'.bmp')
#
#     img = cv2.imread(img_name)
#     label = cv2.imread(label_name,0)
#     weight = cv2.imread(weight_name,0)
#     rows,cols,channels = img.shape
#     img_L1 = contrast_img(img, 1.12, 2)
#     cv2.imwrite(save_path_img + 'L1_{}.jpg'.format(i), img_L1, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     cv2.imwrite(save_path_label + 'L1_{}.bmp'.format(i), label)
#     cv2.imwrite(save_path_weight + 'L1_{}.bmp'.format(i), label)

#     print(i)



#######################################################################
'''平移图片'''
# for i in range(n):
#     i = i
#     temp_list = file_list[i]
#     img_name = os.path.join(train_img_path, temp_list[:-4] + '.jpg')
#     label_name = os.path.join(label_img_path, temp_list[:-4] + '.bmp')
#     weight_name = os.path.join(weight_img_path, temp_list[:-4] + '.bmp')
#
#     img = cv2.imread(img_name)
#     label = cv2.imread(label_name, 0)
#     weight = cv2.imread(weight_name, 0)
#     rows,cols = img.shape[:2]
#
#     # #平移矩阵[[1,0,40],[0,1,45]]
#     M_6 = np.array([[1, 0, 40], [0, 1, 45]], dtype=np.float32)
#     img_M6 = cv2.warpAffine(img, M_6, (rows, cols))
#     label_M6 = cv2.warpAffine(label, M_6, (rows, cols), borderValue=(255,255, 255))
#     weight_M6 = cv2.warpAffine(weight, M_6, (rows, cols), borderValue=(0, 0, 0))
#
#     cv2.imwrite(save_path_img + 'M6_{}.jpg'.format(i), img_M6, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     cv2.imwrite(save_path_label + 'M6_{}.bmp'.format(i), label_M6)
#     cv2.imwrite(save_path_weight + 'M6_{}.bmp'.format(i), weight_M6)
#
#
#     # #平移矩阵[[1,0,20],[0,1,20]]
#     M_5 = np.array([[1, 0, 20], [0, 1, 20]], dtype=np.float32)
#     img_M5 = cv2.warpAffine(img, M_5, (rows, cols))
#     label_M5 = cv2.warpAffine(label, M_5, (rows, cols), borderValue=(255,255, 255))
#     weight_M5 = cv2.warpAffine(weight, M_5, (rows, cols), borderValue=(0, 0, 0))
#
#     cv2.imwrite(save_path_img + 'M5_{}.jpg'.format(i), img_M5, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     cv2.imwrite(save_path_label + 'M5_{}.bmp'.format(i), label_M5)
#     cv2.imwrite(save_path_weight + 'M5_{}.bmp'.format(i), weight_M5)
#
#
#     # #平移矩阵[[1,0,30],[0,1,30]]
#     M_4 = np.array([[1, 0, 30], [0, 1, 30]], dtype=np.float32)
#     img_M4 = cv2.warpAffine(img, M_4, (rows, cols))
#     label_M4 = cv2.warpAffine(label, M_4, (rows, cols), borderValue=(255,255, 255))
#     weight_M4 = cv2.warpAffine(weight, M_4, (rows, cols), borderValue=(0, 0, 0))
#
#     cv2.imwrite(save_path_img + 'M4_{}.jpg'.format(i), img_M4, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     cv2.imwrite(save_path_label + 'M4_{}.bmp'.format(i), label_M4)
#     cv2.imwrite(save_path_weight + 'M4_{}.bmp'.format(i), weight_M4)
#
#
#     # #平移矩阵[[1,0,60],[0,1,60]]
#     M_3=np.array([[1,0,60],[0,1,60]],dtype=np.float32)
#     img_M3 = cv2.warpAffine(img,M_3,(rows,cols))
#     label_M3 = cv2.warpAffine(label,M_3,(rows,cols),borderValue=(255,255, 255))
#     weight_M3 = cv2.warpAffine(weight, M_3, (rows, cols), borderValue=(0, 0, 0))
#
#     cv2.imwrite(save_path_img + 'M3_{}.jpg'.format(i), img_M3, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     cv2.imwrite(save_path_label + 'M3_{}.bmp'.format(i), label_M3)
#     cv2.imwrite(save_path_weight + 'M3_{}.bmp'.format(i), weight_M3)
#
#
#     # #平移矩阵[[1,0,24],[0,1,42]]
#     M_2 = np.array([[1, 0, 24], [0, 1, 42]], dtype=np.float32)
#     img_M2 = cv2.warpAffine(img, M_2, (rows, cols))
#     label_M2 = cv2.warpAffine(label, M_2, (rows, cols), borderValue=(255,255, 255))
#     weight_M2 = cv2.warpAffine(weight, M_2, (rows, cols), borderValue=(0, 0, 0))
#
#     cv2.imwrite(save_path_img + 'M2_{}.jpg'.format(i), img_M2, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     cv2.imwrite(save_path_label + 'M2_{}.bmp'.format(i), label_M2)
#     cv2.imwrite(save_path_weight + 'M2_{}.bmp'.format(i), weight_M2)
#
#
#     # #平移矩阵[[1,0,29],[0,1,18]]
#     M_1 = np.array([[1, 0, 29], [0, 1, 18]], dtype=np.float32)
#     img_M1 = cv2.warpAffine(img, M_1, (rows, cols))
#     label_M1 = cv2.warpAffine(label, M_1, (rows, cols), borderValue=(255,255, 255))
#     weight_M1 = cv2.warpAffine(weight, M_1, (rows, cols), borderValue=(0, 0, 0))
#
#     cv2.imwrite(save_path_img + 'M1_{}.jpg'.format(i), img_M1, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
#     cv2.imwrite(save_path_label + 'M1_{}.bmp'.format(i), label_M1)
#     cv2.imwrite(save_path_weight + 'M1_{}.bmp'.format(i), weight_M1)
#
#
#     print(i)
#
# print('SAVE IMG DONE')
#####################################################################
'''翻转与旋转图片'''
#
# for i in range(n):
#     temp_list = file_list[i]
#     img_name = os.path.join(train_img_path,temp_list[:-4]+'.jpg')
#     label_name = os.path.join(label_img_path,temp_list[:-4]+'.bmp')
#     weight_name = os.path.join(weight_img_path,temp_list[:-4]+'.bmp')
#
#     img = cv2.imread(img_name)
#     label = cv2.imread(label_name,0)
#     weight = cv2.imread(weight_name,0)
#     rows, cols = img.shape[:2]
#
#     #水平镜像
#     img_f1 = cv2.flip(img,1)
#     label_f1 = cv2.flip(label,1)
#     weight_f1 = cv2.flip(weight,1)
#
#     cv2.imwrite(save_path_img+'F_1_{}.jpg'.format(i),img_f1,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#     cv2.imwrite(save_path_label+'F_1_{}.bmp'.format(i),label_f1)
#     cv2.imwrite(save_path_weight+'F_1_{}.bmp'.format(i),weight_f1)
#
#     #垂直镜像
#     img_f0 = cv2.flip(img, 0)
#     label_f0 = cv2.flip(label, 0)
#     weight_f0 = cv2.flip(weight, 0)
#
#     cv2.imwrite(save_path_img + 'F_0_{}.jpg'.format(i), img_f0,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#     cv2.imwrite(save_path_label + 'F_0_{}.bmp'.format(i), label_f0)
#     cv2.imwrite(save_path_weight + 'F_0_{}.bmp'.format(i), weight_f0)
#
#
#     #水平垂直镜像
#     img_f01 = cv2.flip(img, -1)
#     label_f01 = cv2.flip(label, -1)
#     weight_f01 = cv2.flip(weight, -1)
#
#     cv2.imwrite(save_path_img + 'F_-1_{}.jpg'.format(i), img_f01,[int(cv2.IMWRITE_JPEG_QUALITY),100])
#     cv2.imwrite(save_path_label + 'F_-1_{}.bmp'.format(i), label_f01)
#     cv2.imwrite(save_path_weight + 'F_-1_{}.bmp'.format(i), weight_f01)
#

    # #90度旋转
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    # img_90 = cv2.warpAffine(img,M,(cols,rows))
    # label_90 = cv2.warpAffine(label,M,(cols,rows),borderValue=(255, 255, 255))
    # cv2.imwrite(save_path_img + 'R90_{}.jpg'.format(i), img_90,[int(cv2.IMWRITE_JPEG_QUALITY),100])
    # cv2.imwrite(save_path_label + 'R90_{}.bmp'.format(i), label_90)
    #
    # #45度旋转
    # M_1 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
    # img_45 = cv2.warpAffine(img, M_1, (cols, rows))
    # label_45 = cv2.warpAffine(label, M_1, (cols, rows), borderValue=(255, 255, 255))
    # cv2.imwrite(save_path_img + 'R45_{}.jpg'.format(i), img_45,[int(cv2.IMWRITE_JPEG_QUALITY),100])
    # cv2.imwrite(save_path_label + 'R45_{}.bmp'.format(i), label_45)

#     print(i+1)
# print('SAVE IMG DONE')
'''

# # 水平镜像
# h_flip1 = cv2.flip(img1,1)
# h_flip2 = cv2.flip(img2,1)
# cv2.imshow("Flipped Horizontally",h_flip1)
# cv2.imshow('Flipped label',h_flip2)

# #垂直镜像
# v_flip1 = cv2.flip(img1,0)
# v_flip2 = cv2.flip(img2,0)
# cv2.imshow("Flipped Vertically",v_flip1)
# cv2.imshow('Flipped label',v_flip2)

#水平垂直镜像
# hv_flip1 = cv2.flip(img1,-1)
# hv_filp2 = cv2.flip(img2,-1)
# cv2.imshow("Flipped Horizontally & Vertically",hv_flip1)
# cv2.imshow('Flipped label',hv_filp2)
#
#
# #平移矩阵[[1,0,100],[0,1,200]]
# M=np.array([[1,0,200],[0,1,100]],dtype=np.float32)
# img_change1=cv2.warpAffine(img1,M,(rows,cols))
# img_change2=cv2.warpAffine(img2,M,(rows,cols),borderValue=(255, 255, 255))
#
# cv2.imshow("translation",img_change1)
# cv2.imshow('translationlabel',img_change2)


# # #平移矩阵[[1,0,200],[0,1,300]]
# M=np.array([[1,0,200],[0,1,300]],dtype=np.float32)
# img_change1=cv2.warpAffine(img1,M,(rows,cols))
# img_change2=cv2.warpAffine(img2,M,(rows,cols),borderValue=(255, 255, 255))
#
# cv2.imshow("translation",img_change1)
# cv2.imshow('translationlabel',img_change2)

#
# #90度旋转
# rows,cols=img.shape[:2]
# M=cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
# dst1 = cv2.warpAffine(img1,M,(cols,rows))
# dst2 = cv2.warpAffine(img2,M,(cols,rows),borderValue=(255, 255, 255))
# # print(dst1.shape)
# # print(dst2.shape)
# cv2.imshow("90",dst1)
# cv2.imshow('90label',dst2)
#
# #45度旋转
# # rows,cols=img.shape[:2]
# M=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
# dst1=cv2.warpAffine(img1,M,(cols,rows))
# dst2=cv2.warpAffine(img2,M,(cols,rows),borderValue=(255, 255, 255))
#
# cv2.imshow("45",dst1)
# cv2.imshow('45label',dst2)



# cv2.waitKey(0)
'''