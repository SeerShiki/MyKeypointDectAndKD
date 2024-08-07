import numpy as np
import cv2

def adjust_brightness(image, brightness):
    """
        提高/降低亮度
        :param image: 输入图像;
        :param brightness: 增加/降低的亮度 正为增加，负为降低
        :return: 调整后的图像
    """
    if brightness != 0:
        image = image.astype(np.float32)
        image = image + brightness
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
    return image

def sharpen_image(image):
    """
        锐化图像
        :param image: 输入图像;
        :return: 锐化后的图像
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened