# preprocessing.py

import cv2
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from PIL import Image

#
# def normalize_image(img, size=(952, 1360)):
#     """ Normalize the image """
#     img = np.array(img)  # Convert Image object to NumPy array
#     max_r, max_c = size
#     # Apply a Gaussian filter on the image to remove small components
#     blur_radius = 2
#     blurred_image = ndimage.gaussian_filter(img, blur_radius)
#
#     # Binarize the image using OTSU's algorithm
#     threshold, binarized_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # Find the center of mass
#     r, c = np.where(binarized_image == 0)
#     r_center = int(r.mean() - r.min())
#     c_center = int(c.mean() - c.min())
#
#     # Crop the image with a tight box
#     cropped = img[r.min(): r.max(), c.min(): c.max()]
#
#     # Center the image
#     img_r, img_c = cropped.shape
#     normalized_image = np.ones((max_r, max_c), dtype=np.uint8) * 255
#     r_start = max_r // 2 - r_center
#     c_start = max_c // 2 - c_center
#
#     # Make sure the new image does not go off bounds
#     if img_r > max_r:
#         print('Warning: cropping image. The signature should be smaller than the canvas size')
#         r_start = 0
#         difference = img_r - max_r
#         crop_start = difference // 2
#         cropped = cropped[crop_start:crop_start + max_r, :]
#         img_r = max_r
#     else:
#         extra_r = (r_start + img_r) - max_r
#         if extra_r > 0:
#             r_start -= extra_r
#         if r_start < 0:
#             r_start = 0
#
#     if img_c > max_c:
#         print('Warning: cropping image. The signature should be smaller than the canvas size')
#         c_start = 0
#         difference = img_c - max_c
#         crop_start = difference // 2
#         cropped = cropped[:, crop_start:crop_start + max_c]
#         img_c = max_c
#     else:
#         extra_c = (c_start + img_c) - max_c
#         if extra_c > 0:
#             c_start -= extra_c
#         if c_start < 0:
#             c_start = 0
#
#     # Add the image to the blank canvas
#     normalized_image[r_start:r_start + img_r, c_start:c_start + img_c] = cropped
#
#     # Remove noise above the threshold
#     normalized_image[normalized_image > threshold] = 255
#
#     return normalized_image
#
#
# def resize_image(image, new_size, interpolation='bilinear'):
#     height, width = new_size
#
#     # Resize the image
#     image = resize(image.astype(np.float32), (height, width), mode='reflect')
#
#     # Crop to exactly the desired new_size, using the middle of the image
#     start_y = (image.shape[0] - height) // 2
#     start_x = (image.shape[1] - width) // 2
#     cropped = image[start_y: start_y + height, start_x:start_x + width]
#
#     return cropped
#
#
# def crop_center(img, input_shape):
#     img_shape = img.shape
#     start_y = (img_shape[0] - input_shape[0]) // 2
#     start_x = (img_shape[1] - input_shape[1]) // 2
#     cropped = img[start_y: start_y + input_shape[0], start_x:start_x + input_shape[1]]
#     return cropped
# preprocessing.py

import cv2
import numpy as np


def preprocess_image(image_data, target_size=(224, 224)):
    try:
        # Decode image data
        nparr = np.fromstring(image_data.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize image
        image = cv2.resize(image, target_size)

        # Normalize pixel values
        image = image / 255.0

        return image
    except Exception as e:
        print("Error preprocessing image")
        print(e)
        return None
