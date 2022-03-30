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
from PIL import Image
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

def NormalizeData(data):
    """
    return the array normalized to numbers between 0 and 1
    :param data:
    :return:
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    if representation == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (img - img.min()) / (img.max() - img.min())
    else:  # we should represent in RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (img - img.min()) / (img.max() - img.min())
    # image = cv2.imread(filename)
    # if representation == 1:
    #     im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     # img_gray = NormalizeData(img_gray)
    #     norm_image = cv2.normalize(im_gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #     return norm_image
    # else:
    #     im_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     norm_image = cv2.normalize(im_RGB, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #     return norm_image


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    plt.imshow(img)
    plt.gray()  # to change the view of the picture to actual greyscale
    plt.show()
    # img = cv2.imread(filename)
    # if representation == 2:
    #     im_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     plt.imshow(im_RGB)
    #     plt.show()
    # else:
    #     plt.imshow(img,cmap='gray')
    #     plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    YIQ_from_RGB = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])

    YIQ = np.dot(imgRGB,YIQ_from_RGB.transpose())
    return YIQ

def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    YIQ_from_RGB = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    YIQ_conversion_inverse = np.linalg.inv(YIQ_from_RGB)
    imgRGB = np.dot(imgYIQ, YIQ_conversion_inverse.transpose())
    return imgRGB

def cumSum_calc(arr: np.array) -> np.ndarray:
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
    if len(imgOrig.shape) == 2:
        return histEq2Shape(imgOrig)
    else:
        yi = transformRGB2YIQ(imgOrig)
        y = yi[:,:,0]
        imEq, histOrig255, histEq = histEq2Shape(y)
        yi[:, :, 0] = imEq
        imE = transformYIQ2RGB(yi)
        return imE, histOrig255, histEq




def histEq2Shape(imgOrig):
    orig255 = cv2.normalize(imgOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    orig255 = orig255.astype(np.uint8)
    histOrig255 = np.zeros(256)
    for pix in range(256):
        histOrig255[pix] = np.count_nonzero(orig255 == pix)
    cs = cumSum_calc(histOrig255)
    imEq = np.zeros_like(imgOrig, dtype=float)
    for i in range(256):
        imEq[orig255 == i] = int(cs[i])
    histEq = np.zeros(256)
    for pix in range(256):
        histEq[pix] = np.count_nonzero(imEq == pix)
    imEq = imEq / 255.0

    return imEq, histOrig255, histEq

def find_q(z: np.array, image_hist: np.ndarray) -> np.ndarray:
    """
        Calculate the new q using wighted average on the histogram
        :param image_hist: the histogram of the original image
        :param z: the new list of centers
        :return: the new list of wighted average
    """
    q = [np.average(np.arange(z[k], z[k + 1] + 1), weights=image_hist[z[k]: z[k + 1] + 1]) for k in range(len(z) - 1)]
    return np.round(q).astype(int)


def find_new_z(q: np.array) -> np.array:
    """
        Calculate the new z using the formula from the lecture.
        :param q: the new list of q
        :param z: the old z
        :return: the new z
    """
    z_new = np.array([round((q[i - 1] + q[i]) / 2) for i in range(1, len(q))]).astype(int)
    z_new = np.concatenate(([0], z_new, [255]))
    return z_new

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    if len(imOrig.shape) == 2:  # single channel (grey channel)
        flag = 2
        return Quant2Shape(imOrig.copy(), nQuant, nIter, flag, 0)


    flag = 3
    yiqImg = transformRGB2YIQ(imOrig)
    return Quant2Shape(yiqImg[:, :, 0].copy(), nQuant, nIter, flag, yiqImg)  # y channel = yiqImg[:, :, 0].copy()
    # qImage = []
    # for img in qImage_:
    #     # convert the original img back from YIQ to RGB
    #     qImage_i = transformYIQ2RGB(np.dstack((img, yiqImg[:, :, 1], yiqImg[:, :, 2])))
    #     qImage.append(qImage_i)
    #
    # return qImage, mse

def Quant2Shape(imOrig, nQuant, nIter, flag, img):
    Q_IM = []
    MSE = []
    orig255 = cv2.normalize(imOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    orig255 = orig255.astype(np.uint8)
    histOrig255 = np.zeros(256)
    for pix in range(256):
        histOrig255[pix] = np.count_nonzero(orig255 == pix)
    z_border = np.zeros(nQuant + 1, dtype=int)
    for i in range(nQuant + 1):
        z_border[i] = i * (255 / nQuant)
    for i in range(nIter):
        q = find_q(z_border, histOrig255)
        qImage_i = np.zeros_like(imOrig)
        for k in range(len(q)):
            qImage_i[orig255 > z_border[k]] = q[k]

        z_border = find_new_z(q)
        MSE.append(np.sqrt((orig255 - qImage_i) ** 2).mean())
        Q_IM.append(qImage_i / 255.0)
    if flag == 3:
        qImage = []
        for i in range(len(Q_IM)):
            img[:, :, 0] = Q_IM[i]
            imE = transformYIQ2RGB(img)
            qImage.append(imE)
            # Q_IM[i][Q_IM[i] > 1] = 1
            # Q_IM[i][Q_IM[i] < 0] = 0
        return qImage,MSE
    return Q_IM, MSE
