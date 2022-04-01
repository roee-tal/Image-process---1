"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import matplotlib.pyplot as plt
import cv2
import numpy as np
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 315858506


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    try:
        img = cv2.imread(filename)
        if representation == LOAD_GRAY_SCALE:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return (img - img.min()) / (img.max() - img.min())
        else:  # rgb
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return (img - img.min()) / (img.max() - img.min())
    except:
        print("Error")

def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    try:
        img = imReadAndConvert(filename, representation)
        plt.imshow(img)
        plt.gray()  # to change the image to grayscale
        plt.show()
    except:
        print("Error")

def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    try:
        YIQ_from_RGB = np.array([[0.299, 0.587, 0.114],
                                 [0.59590059, -0.27455667, -0.32134392],
                                 [0.21153661, -0.52273617, 0.31119955]])

        YIQ = np.dot(imgRGB,YIQ_from_RGB.transpose().copy())
        return YIQ
    except:
        print("Error")


def f_between0_1(rgb_img,imgRGB):

    rgb_img[:, :, 0] = (imgRGB[:, :, 0] - np.min(imgRGB[:, :, 0])) / (np.max(imgRGB[:, :, 0]) - np.min(imgRGB[:, :, 0]))
    rgb_img[:, :, 1] = (imgRGB[:, :, 1] - np.min(imgRGB[:, :, 1])) / (np.max(imgRGB[:, :, 1]) - np.min(imgRGB[:, :, 1]))
    rgb_img[:, :, 2] = (imgRGB[:, :, 2] - np.min(imgRGB[:, :, 2])) / (np.max(imgRGB[:, :, 2]) - np.min(imgRGB[:, :, 2]))

    return rgb_img


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    try:
        YIQ_from_RGB = np.array([[0.299, 0.587, 0.114],
                                 [0.59590059, -0.27455667, -0.32134392],
                                 [0.21153661, -0.52273617, 0.31119955]])
        YIQ_conversion_inverse = np.linalg.inv(YIQ_from_RGB)
        imgRGB = np.dot(imgYIQ, YIQ_conversion_inverse.transpose().copy())
        rgb_img = imgYIQ.copy()
        imgRG = f_between0_1(rgb_img,imgRGB) # To be in range [0..1] for float
        return imgRG
    except:
        print("Error")

def cumSum_calc(arr: np.array) -> np.ndarray:
    """
        help function to calculate the cumsum (want to try it by myself)
        :param arr: Original Histogram
        :return: the cumsum
    """
    cum_sum = np.zeros_like(arr)
    cum_sum[0] = arr[0]
    arr_len = len(arr)
    for idx in range(1, arr_len):
        cum_sum[idx] = arr[idx] + cum_sum[idx - 1]
    n_cum_sum = ((cum_sum - cum_sum.min()) * 255)/(cum_sum.max() - cum_sum.min())
    n_cum_sum = n_cum_sum.astype('uint8')
    return n_cum_sum


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    try:
        if len(imgOrig.shape) == 2:  # If the pic is grayscale
            return histEq2Shape(imgOrig)
        else:  # If the pic is rgb -  operate on the Y channel of YIQ, and the convert back to grayscale
            yi = transformRGB2YIQ(imgOrig)
            y = yi[:,:,0]
            imEq, histOrig255, histEq = histEq2Shape(y)
            yi[:, :, 0] = imEq
            imE = transformYIQ2RGB(yi)
            return imE, histOrig255, histEq
    except:
        print("error")



def histEq2Shape(imgOrig):
    """
        help function to equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: the new pic, the origin histogram, the new histogram
    """
    # normalize from 0-1 to 0-255
    orig255 = cv2.normalize(imgOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    orig255 = orig255.astype(np.uint8)
    # calculate the histogram
    histOrig255 = np.zeros(256)
    for pix in range(256):
        histOrig255[pix] = np.count_nonzero(orig255 == pix)
    # calculate the normalized cumsum with help function + LUT
    cs = cumSum_calc(histOrig255)
    # replace each intensity with LUT[i]
    imEq = np.zeros_like(imgOrig, dtype=float)
    for i in range(256):
        imEq[orig255 == i] = int(cs[i])
    # calculate the new histogram
    histEq = np.zeros(256)
    for pix in range(256):
        histEq[pix] = np.count_nonzero(imEq == pix)
    # Normalize back to 0-1
    imEq = imEq / 255.0

    return imEq, histOrig255, histEq

def find_q(z: np.array, image_hist: np.ndarray) -> np.ndarray:
    """
        Help function to calculate the new q values for each border
        :param image_hist: the histogram of the original image
        :param z: border's list
        :return: the new list of q
    """
    q = [np.average(np.arange(z[k], z[k + 1] + 1), weights=image_hist[z[k]: z[k + 1] + 1]) for k in range(len(z) - 1)]
    return np.round(q).astype(int)


def find_new_z(q: np.array) -> np.array:
    """
        Help function to calculate the new borders using the formula from the lecture.
        :param q: the new list of q
        :return: the new borders
    """
    z_new = np.array([round((q[i - 1] + q[i]) / 2) for i in range(1, len(q))]).astype(int)
    z_new = np.concatenate(([0], z_new, [255]))
    return z_new

def find_first_z(pixel_num, nQuant, histOrig):
    """
         Help function to find the first borders - according to the pixel's amount - same amount in each part
        :param pixel_num: pixel's amount
        :param nQuant: Number of colors to quantize the image to
        :param histOrig: The original histogram
        :return: The first borders
    """
    # In order to put same amount in each part - i used the cumsum
    cumsum = np.cumsum(histOrig)
    new_z =np.zeros(nQuant + 1, dtype=int)
    new_z[0] = 0
    new_z[len(new_z) - 1] = 255
    bound1 = pixel_num / nQuant
    bound = bound1
    i = 1
    for x in range(255):
        # check which gray rank is crossing the bound, for each bound
        if (cumsum[x] >= bound):
            new_z[i] = x
            i = i + 1
            bound = bound + bound1
    new_z = new_z.astype(int)
    return new_z

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    try:
        if len(imOrig.shape) == 2: # grayscale
            flag = 2
            return Quant2Shape(imOrig.copy(), nQuant, nIter, flag, 0)

        flag = 3 # rgb - operate on the Y channel of YIQ
        yiqImg = transformRGB2YIQ(imOrig)
        return Quant2Shape(yiqImg[:, :, 0].copy(), nQuant, nIter, flag, yiqImg)  # y channel = yiqImg[:, :, 0].copy()
    except:
        print("Error")

def Quant2Shape(imOrig, nQuant, nIter, flag, img):
    """
        Help function to quantize an image with shape 2 in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :param flag: the shape of the image
        :param img: The whole img(relevant for rgb case)
        :return: (List[qImage_i],List[error_i])
    """
    Q_IM = []
    MSE = []
    # normalize from 0-1 to 0-255
    orig255 = cv2.normalize(imOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    orig255 = orig255.astype(np.uint8)
    # calculate the histogram
    histOrig255 = np.zeros(256)
    for pix in range(256):
        histOrig255[pix] = np.count_nonzero(orig255 == pix)
    pixel_num = (float)(imOrig.shape[0] * imOrig.shape[1])
    # find the first z values
    z_border = find_first_z(pixel_num,nQuant,histOrig255)
    for i in range(nIter):
        # find the new q values
        q = find_q(z_border, histOrig255)
        qImage_i = np.zeros_like(imOrig)
        # update the color to be the mean color for each part
        for k in range(len(q)):
            qImage_i[orig255 > z_border[k]] = q[k]
        z_border = find_new_z(q)
        # calculate mse
        MSE.append(np.sqrt((orig255 - qImage_i) ** 2).mean())
        # add the normalized [0,1] pic to the list
        Q_IM.append(qImage_i / 255.0)
    # In case of rgb - transform back to rgb from yiq
    if flag == 3:
        for i in range(len(Q_IM)):
            img[:, :, 0] = Q_IM[i]
            Q_IM[i] = transformYIQ2RGB(img)
            Q_IM[i][Q_IM[i] > 1] = 1
            Q_IM[i][Q_IM[i] < 0] = 0
    return Q_IM, MSE


