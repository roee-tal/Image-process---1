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
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np

title_window = 'Gamma Correction'

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    try:
        # Global variable - need for using in the help function
        global image
        if rep == 2:
            image = cv2.imread(img_path,2)
        else:
            image = cv2.imread(img_path,1)
        cv2.namedWindow(title_window)
        trackbar_name = 'Gamma %d' % 200
        cv2.createTrackbar(trackbar_name, title_window, 100, 200, on_trackbar)
        on_trackbar(100)
        cv2.waitKey()
    except:
        print("error")

def on_trackbar(bright):
    gamma = float(bright) / 100
    # The first slider will be white because invGamma is 0 first
    invGamma = 0
    if gamma != 0:
        invGamma = 1.0 / gamma
    gammaTable = np.array([((i / float(255)) ** invGamma) * 255 for i in np.arange(0, 255 + 1)]).astype("uint8")
    # Lut for the image according to te gamma table
    img = cv2.LUT(image, gammaTable)
    cv2.imshow(title_window, img)


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
