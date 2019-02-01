import cv2


def resize_coeff(x, new_x):
    """
    Evaluate resize coefficient from image shape
    Args:
        x: original value
        new_x: expect value

    Returns:
        Resize coefficient
    """
    return new_x / x


def resize_image(img, resize_shape=(128, 128), interpolation=cv2.INTER_AREA):
    """
    Resize single image
    Args:
        img: input image
        resize_shape: resize shape in format (height, width)
        interpolation: interpolation method

    Returns:
        Resized image
    """
    return cv2.resize(img, None, fx=resize_coeff(img.shape[1], resize_shape[1]),
                     fy=resize_coeff(img.shape[0], resize_shape[0]),
                     interpolation=interpolation)