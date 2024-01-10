import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Apply the frequency filter to an image, removing the noise
"""


def remove_noise_with_frequency_filter(image_path, output_path, filter_type, filter_size=30, band_lower=10,
                                       band_upper=50):
    # 1) perform the 2D fourier transform
    fourier_transform = np.fft.fft2(image_path)
    fourier_transform_shifted = np.fft.fftshift(fourier_transform)

    # 2) obtain image dimension
    width, height = image_path.shape
    center_width, center_height = width // 2, height // 2

    # 3) create a mask with ones only
    mask = np.ones((width, height), np.uint8)

    # 4) create the different filters, calculating the rows and columns at the center, returning the mask
    if filter_type == 'low_pass':
        mask[center_width - filter_size:center_width + filter_size,
        center_height - filter_size:center_height + filter_size] = 0
    elif filter_type == 'high_pass':
        mask[center_width - filter_size:center_width + filter_size,
        center_height - filter_size:center_height + filter_size] = 1
    elif filter_type == 'band_pass':
        mask[center_width - band_upper:center_width + band_upper,
        center_width - band_upper:center_width + band_upper] = 0
        mask[center_width - band_lower:center_width + band_lower,
        center_width - band_lower:center_width + band_lower] = 1
    elif filter_type == 'band_reject':
        mask[center_width - band_upper:center_width + band_upper,
        center_width - band_upper:center_width + band_upper] = 1
        mask[center_width - band_lower:center_width + band_lower,
        center_width - band_lower:center_width + band_lower] = 0

    # 5) obtained the mask, multiply the fourier transform with the msak
    fourier_mask_shifted_filtered = fourier_transform_shifted * mask

    # 6) apply the inverse fourier to obtain the image filtered
    image_filtered = np.fft.ifft2(np.fft.ifftshift(fourier_mask_shifted_filtered)).real

    # 7) save the filtered image  with the uint8 format
    image_filtered_uint8 = np.uint8(image_filtered)

    cv2.imwrite(output_path, image_filtered_uint8)

    return image_filtered_uint8


"""
Removes salt and pepper noise from an image, returning the enhanced version of it
"""


def remove_salt_and_pepper_noise_from_images(image_path, output_path, kernel_size=3):
    filtered_img = cv2.medianBlur(image_path, kernel_size)
    cv2.imwrite(output_path, filtered_img)
    return filtered_img


"""
Removes the blur effect from an image, returning the enhanced version of it
"""


def remove_blurred_from_images(image_path, output_path):
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen = cv2.filter2D(image_path, -1, sharpen_kernel)
    cv2.imwrite(output_path, sharpen)
    return sharpen


"""
Removes the gaussian effect from an image, returning the enhanced version of it
"""


def remove_gaussian_noise_from_images(image_path, output_path, kernel_size=5):
    gaussian_noise = cv2.GaussianBlur(image_path, (kernel_size, kernel_size), 0)
    cv2.imwrite(output_path, gaussian_noise)
    return gaussian_noise


def display_without_salt_pepper_noise(image, path_to_save):
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(remove_salt_and_pepper_noise_from_images(image, path_to_save), cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()


def display_without_blur_effect(image, path_to_save):
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(remove_blurred_from_images(image, path_to_save), cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()


def display_without_gaussian_noise_effect(image, path_to_save):
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(remove_gaussian_noise_from_images(image, path_to_save), cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()


def display_spatial_filter_type(image, path_to_save, filter_type):
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(remove_noise_with_frequency_filter(image, path_to_save, filter_type), cmap='gray')
    plt.title(f'Frequency filter{filter_type} Enhanced image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()


if __name__ == "__main__":
    image_face = cv2.imread('Dataset/GrayscaleNoisyImages/face.bmp')
    display_without_gaussian_noise_effect(image_face, 'SavedImages/GrayscaleNoisyImages_face.png')

    image_face_1 = cv2.imread('Dataset/GrayscaleNoisyImages/face_1.bmp')
    display_without_gaussian_noise_effect(image_face, 'SavedImages/GrayscaleNoisyImages_face_1.png')

    image_face_2 = cv2.imread('Dataset/GrayscaleNoisyImages/face_2.bmp')
    display_without_gaussian_noise_effect(image_face, 'SavedImages/GrayscaleNoisyImages_face_2.png')

    image_face_3 = cv2.imread('Dataset/GrayscaleNoisyImages/face_3.bmp')
    display_without_gaussian_noise_effect(image_face, 'SavedImages/GrayscaleNoisyImages_face_3.png')

    image_face_4 = cv2.imread('Dataset/GrayscaleNoisyImages/face_4.bmp')
    display_without_gaussian_noise_effect(image_face, 'SavedImages/GrayscaleNoisyImages_face_4.png')

    image_face_5 = cv2.imread('Dataset/GrayscaleNoisyImages/face_5.bmp')
    display_without_gaussian_noise_effect(image_face_5, 'SavedImages/GrayscaleNoisyImages_face_5.png')

    image_face_thermogram = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram.bmp',cv2.IMREAD_GRAYSCALE)
    display_without_salt_pepper_noise(image_face_thermogram, 'SavedImages/GrayscaleNoisyImages_face_thermogram.png')

    image_face_thermogram_1 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_1.bmp',cv2.IMREAD_GRAYSCALE)
    display_without_salt_pepper_noise(image_face_thermogram_1, 'SavedImages/GrayscaleNoisyImages_face_thermogram_1.png')

    image_face_thermogram_2 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_2.bmp',cv2.IMREAD_GRAYSCALE)
    display_without_salt_pepper_noise(image_face_thermogram_2, 'SavedImages/GrayscaleNoisyImages_face_thermogram_2.png')

    image_face_thermogram_3 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_3.bmp',cv2.IMREAD_GRAYSCALE)
    display_without_salt_pepper_noise(image_face_thermogram_3, 'SavedImages/GrayscaleNoisyImages_face_thermogram_3.png')

    image_face_thermogram_4 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_4.bmp',cv2.IMREAD_GRAYSCALE)
    display_without_salt_pepper_noise(image_face_thermogram_4, 'SavedImages/GrayscaleNoisyImages_face.png_thermogram_4.png')

    image_face_thermogram_5 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_5.bmp',cv2.IMREAD_GRAYSCALE)
    display_without_salt_pepper_noise(image_face_thermogram_5,'SavedImages/GrayscaleNoisyImages_face.png_thermogram_5.png')

    image_face_finger = cv2.imread('Dataset/GrayscaleNoisyImages/finger.bmp', cv2.IMREAD_GRAYSCALE)
    display_spatial_filter_type(image_face_finger, 'SavedImages/GrayscaleNoisyImages_face_finger_high_pass.png', 'high_pass')
    display_spatial_filter_type(image_face_finger, 'SavedImages/GrayscaleNoisyImages_face_finger_low_pass.png', 'low_pass')
    display_spatial_filter_type(image_face_finger, 'SavedImages/GrayscaleNoisyImages_face_finger_band_pass.png', 'band_pass')
    display_spatial_filter_type(image_face_finger, 'SavedImages/GrayscaleNoisyImages_face_finger_band_reject.png', 'band_reject')

    image_face_finger_1 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_1.bmp', cv2.IMREAD_GRAYSCALE)
    display_spatial_filter_type(image_face_finger_1, 'SavedImages/GrayscaleNoisyImages_face_finger_1_high_pass.png','high_pass')
    display_spatial_filter_type(image_face_finger_1, 'SavedImages/GrayscaleNoisyImages_face_finger_1_low_pass.png','low_pass')
    display_spatial_filter_type(image_face_finger_1, 'SavedImages/GrayscaleNoisyImages_face_finger_1_band_pass.png','band_pass')
    display_spatial_filter_type(image_face_finger_1, 'SavedImages/GrayscaleNoisyImages_face_finger_1_band_reject.png','band_reject')

    image_face_finger_2 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_2.bmp', cv2.IMREAD_GRAYSCALE)
    display_spatial_filter_type(image_face_finger_2, 'SavedImages/GrayscaleNoisyImages_face_finger_2_high_pass.png','high_pass')
    display_spatial_filter_type(image_face_finger_2, 'SavedImages/GrayscaleNoisyImages_face_finger_2_low_pass.png','low_pass')
    display_spatial_filter_type(image_face_finger_2, 'SavedImages/GrayscaleNoisyImages_face_finger_2_band_pass.png','band_pass')
    display_spatial_filter_type(image_face_finger_2, 'SavedImages/GrayscaleNoisyImages_face_finger_2_band_reject.png','band_reject')

    image_face_finger_3 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_3.bmp', cv2.IMREAD_GRAYSCALE)
    display_spatial_filter_type(image_face_finger_3, 'SavedImages/GrayscaleNoisyImages_face_finger_3_high_pass.png','high_pass')
    display_spatial_filter_type(image_face_finger_3, 'SavedImages/GrayscaleNoisyImages_face_finger_3_low_pass.png', 'low_pass')
    display_spatial_filter_type(image_face_finger_3, 'SavedImages/GrayscaleNoisyImages_face_finger_3_band_pass.png', 'band_pass')
    display_spatial_filter_type(image_face_finger_3, 'SavedImages/GrayscaleNoisyImages_face_finger_3_band_reject.png', 'band_reject')

    image_face_finger_4 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_4.bmp', cv2.IMREAD_GRAYSCALE)
    display_spatial_filter_type(image_face_finger_4, 'SavedImages/GrayscaleNoisyImages_face_finger_4.png','high_pass')
    display_spatial_filter_type(image_face_finger_4, 'SavedImages/GrayscaleNoisyImages_face_finger_4.png','low_pass')
    display_spatial_filter_type(image_face_finger_4, 'SavedImages/GrayscaleNoisyImages_face_finger_4.png', 'band_pass')
    display_spatial_filter_type(image_face_finger_4, 'SavedImages/GrayscaleNoisyImages_face_finger_4.png','band_reject')

    image_face_finger_5 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_5.bmp', cv2.IMREAD_GRAYSCALE)
    display_spatial_filter_type(image_face_finger_5, 'SavedImages/GrayscaleNoisyImages_face_finger_5.png', 'high_pass')
    display_spatial_filter_type(image_face_finger_5, 'SavedImages/GrayscaleNoisyImages_face_finger_5.png', 'low_pass')
    display_spatial_filter_type(image_face_finger_5, 'SavedImages/GrayscaleNoisyImages_face_finger_5.png', 'band_pass')
    display_spatial_filter_type(image_face_finger_5, 'SavedImages/GrayscaleNoisyImages_face_finger_5.png','band_reject')

    image_face_iris = cv2.imread('Dataset/GrayscaleNoisyImages/iris.bmp',cv2.IMREAD_GRAYSCALE)
    display_without_blur_effect(image_face_iris, 'SavedImages/GrayscaleNoisyImages_face_iris.png')

    image_face_iris_1 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_1.bmp',cv2.IMREAD_GRAYSCALE)
    display_without_blur_effect(image_face_iris_1, 'SavedImages/GrayscaleNoisyImages_face_iris_1.png')

    image_face_iris_2 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_2.bmp',cv2.IMREAD_GRAYSCALE)
    display_without_blur_effect(image_face_iris_2, 'SavedImages/GrayscaleNoisyImages_face_iris_2.png')

    image_face_iris_3 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_3.bmp',cv2.IMREAD_GRAYSCALE)
    display_without_blur_effect(image_face_iris_3, 'SavedImages/GrayscaleNoisyImages_face_iris_3.png')

    image_face_iris_4 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_4.bmp',cv2.IMREAD_GRAYSCALE)
    display_without_blur_effect(image_face_iris_4, 'SavedImages/GrayscaleNoisyImages_face_iris_4.png')

    image_face_iris_5 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_5.bmp',cv2.IMREAD_GRAYSCALE)
    display_without_blur_effect(image_face_iris_5, 'SavedImages/GrayscaleNoisyImages_face_iris_5.png')
