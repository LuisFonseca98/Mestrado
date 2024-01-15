import cv2
import numpy as np
import matplotlib.pyplot as plt


#    Returns mean squared error between 'mat1' and 'mat2'
def meanSquaredError(mat1, mat2):
    return np.square(mat1 - mat2).mean()


#    Returns mean absolute error between 'mat1' and 'mat2'
def meanAbsoluteError(mat1, mat2):
    return np.absolute(mat1 - mat2).mean()


#    Returns signal to noise ratio given a signal and a noisy signal
def signalToNoiseRatio(signal, noisySignal):
    return 10.0 * np.log10(np.square(signal).sum() / np.square(signal - noisySignal).sum())


def remove_noise_with_frequency_filter(image_path, output_path, filter_type, filter_size=10, band_lower=-1,
                                       band_upper=1):
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
Removes salt and pepper noise from RGB images
"""


def remove_salt_and_pepper_noise_from_rgb_images(image_path, output_path, path_to_save, kernel_size=5):
    filtered_img = cv2.medianBlur(image_path, kernel_size)

    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()
    cv2.imwrite(output_path, filtered_img)
    return cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)


def apply_color_map_to_image(image_path, output_path, path_to_save, color_map_type):
    global pseudocolor_image

    if color_map_type == 'ocean':
        pseudocolor_image = cv2.applyColorMap(image_path, cv2.COLORMAP_OCEAN)

    elif color_map_type == 'hsv':
        pseudocolor_image = cv2.applyColorMap(image_path, cv2.COLORMAP_HSV)

    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(pseudocolor_image, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()

    cv2.imwrite(output_path, pseudocolor_image)
    return cv2.cvtColor(pseudocolor_image, cv2.COLOR_BGR2RGB)


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


def fix_magenta_color_rgb(image_path, output_path, path_to_save, brightness_value=5):
    # calculate the mean for each channel
    mean_b = np.mean(image_path[:, :, 0])
    mean_g = np.mean(image_path[:, :, 1])
    mean_r = np.mean(image_path[:, :, 2])

    # calculate the scalling factors
    scale_b = mean_g / mean_b
    scale_r = mean_g / mean_r

    # after scale, we apply the scaling factors to the blue and red channels
    image_path[:, :, 0] = np.clip((image_path[:, :, 0].astype(np.float64) * scale_b), 0, 255).astype(np.uint8)
    image_path[:, :, 2] = np.clip((image_path[:, :, 2].astype(np.float64) * scale_r), 0, 255).astype(np.uint8)

    # because after scale, we lose lum, we apply some brightness to the image
    image_enhanced = cv2.convertScaleAbs(image_path, alpha=brightness_value, beta=0)

    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()

    cv2.imwrite(output_path, image_enhanced)

    return image_enhanced


"""
Increases the brightness of an image, returning the enhanced version of it
"""


def increase_brightness(image_path, output_path, path_to_save, factor=1.5):
    brighter_image = np.clip(image_path * factor, 0, 255).astype(np.uint8)

    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(image_path)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(brighter_image)
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()

    cv2.imwrite(output_path, cv2.cvtColor(brighter_image, cv2.COLOR_BGR2RGB))

    return brighter_image


def increase_contrast(image_path, output_path, path_to_save, luminance_factor=0.1):
    # Convert BGR to HSV
    hsv_img = cv2.cvtColor(image_path, cv2.COLOR_BGR2HSV)

    # Reduce luminance in the Value channel
    hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] * luminance_factor, 0, 255)

    # Convert HSV back to BGR
    reduced_luminance_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(image_path)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_path)
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()

    cv2.imwrite(output_path, cv2.cvtColor(reduced_luminance_img, cv2.COLOR_BGR2RGB))

    return reduced_luminance_img


"""
Giving an negative image and red noise, we apply filters to remove the noise
and obtain a new image, similar to the original one
"""


def remove_red_noise_rgb_image(image_path, output_path, path_to_save, noise_kernel_size=5):
    # invert the colors of the image, and for that we do a bitwise operation
    positive_image = cv2.bitwise_not(image_path)

    # we define a threshold of the color red(lower and upper)
    lower_red = np.array([100, 0, 0])
    upper_red = np.array([100, 100, 255])

    # apply a mask to the red component
    red_mask = cv2.inRange(positive_image, lower_red, upper_red)

    # reset the red pixels of the original image
    image_without_red_noise = positive_image.copy()
    image_without_red_noise[red_mask != 0] = [255, 255, 255]

    # afterwards we apply a median blur filter to the image, removing the noise
    image_without_red_noise = cv2.medianBlur(image_without_red_noise, noise_kernel_size)

    # plot the results
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_without_red_noise, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()

    # save the image result
    cv2.imwrite(output_path, image_without_red_noise)

    return image_without_red_noise


"""
Removes the red background from the thermal image(face5.png)
"""


def remove_red_background_from_thermal_image(image_path, output_path, path_to_save):
    # Define the lower and upper bounds for the red color in RGB
    lower_red = np.array([0, 0, 100], dtype=np.uint8)
    upper_red = np.array([100, 100, 255], dtype=np.uint8)

    # Create a binary mask based on the red color range
    mask = cv2.inRange(image_path, lower_red, upper_red)

    # Invert the mask to get the regions with black background
    mask_inverted = cv2.bitwise_not(mask)

    # Create a copy of the image with black background
    image_result = image_path.copy()
    image_result[mask > 0] = [0, 0, 0]

    # plot the results
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()

    # save the image result
    cv2.imwrite(output_path, image_result)


"""
Remove the blue component, returning an enhanced image without the blue component
"""


def fix_blue_color_rgb(image_path, output_path, path_to_save):
    # Convert the image to HSV format
    hsv_img = cv2.cvtColor(image_path, cv2.COLOR_BGR2HSV)

    # Set saturation channel to zero
    hsv_img[:, :, 1] = 0

    # Convert the HSV image back to BGR format
    img_without_color = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    # plot the results
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_without_color)
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()

    # save the image result
    cv2.imwrite(output_path, cv2.cvtColor(img_without_color, cv2.COLOR_BGR2RGB))

    return image_path


"""
Removes the bluir effect and enhances the contrast of an image
"""


def remove_blur_and_enhance_contrast(image_path, output_path, path_to_save, contrast_factor=1):
    # Convert the image to LAB color space
    convert_lab_img = cv2.cvtColor(image_path, cv2.COLOR_BGR2Lab)

    # Enhance contrast using the CLAHE (Contrast Limited Adaptive Histogram Equalization) algorithm
    clahe_method = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe_method.apply(convert_lab_img[:, :, 0])

    # Merge the enhanced channel with the original A and B channels
    image_result = cv2.merge([img_enhanced, convert_lab_img[:, :, 1], convert_lab_img[:, :, 2]])

    # Convert the LAB image back to BGR color space
    img_result_bgr = cv2.cvtColor(image_result, cv2.COLOR_Lab2BGR)

    # Enhance overall contrast
    img_result_bgr = cv2.convertScaleAbs(img_result_bgr, alpha=contrast_factor, beta=0)

    # plot the results
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_result_bgr, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()

    # save the image result
    cv2.imwrite(output_path, img_result_bgr)

    return img_result_bgr


def display_without_salt_pepper_noise(image, output_name, path_to_save):
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(remove_salt_and_pepper_noise_from_images(image, output_name), cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()


def display_without_blur_effect(image, output_name, path_to_save):
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(remove_blurred_from_images(image, output_name), cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()


def display_without_gaussian_noise_effect(image, output_name, path_to_save):
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(remove_gaussian_noise_from_images(image, output_name), cmap='gray')
    plt.title('Enhanced Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()


def display_spatial_filter_type(image, output_name, path_to_save, filter_type):
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(remove_noise_with_frequency_filter(image, output_name, filter_type), cmap='gray')
    plt.title(f'Frequency filter{filter_type} Enhanced image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(path_to_save)
    plt.show()


if __name__ == "__main__":
    ##########################################INIT VARIABLES GRAYSCALE PICS#############################################
    # image_face = cv2.imread('Dataset/GrayscaleNoisyImages/face.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_gaussian_noise_effect(image_face, 'SavedImages/GrayscaleNoisyImages_face.png', 'SavedImages/Plot_GrayscaleNoisyImages_face.png')
    # image_face_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face.png', cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face: ', meanSquaredError(image_face, image_face_enhanced))
    # print('MAE on image_face: ', meanAbsoluteError(image_face, image_face_enhanced))
    # print('SNR on image_face: ', signalToNoiseRatio(image_face, image_face_enhanced))
    # print("")
    #
    # image_face_1 = cv2.imread('Dataset/GrayscaleNoisyImages/face_1.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_gaussian_noise_effect(image_face_1, 'SavedImages/GrayscaleNoisyImages_face1.png', 'SavedImages/Plot_GrayscaleNoisyImages_face1.png')
    # image_face_enhanced_1 = cv2.imread('SavedImages/GrayscaleNoisyImages_face1.png', cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_1: ', meanSquaredError(image_face_1, image_face_enhanced_1))
    # print('MAE on image_face_1: ', meanAbsoluteError(image_face_1, image_face_enhanced_1))
    # print('SNR on image_face_1: ', signalToNoiseRatio(image_face_1, image_face_enhanced_1))
    # print("")
    #
    # image_face_2 = cv2.imread('Dataset/GrayscaleNoisyImages/face_2.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_gaussian_noise_effect(image_face_2, 'SavedImages/GrayscaleNoisyImages_face2.png', 'SavedImages/Plot_GrayscaleNoisyImages_face2.png')
    # image_face_enhanced_2 = cv2.imread('SavedImages/GrayscaleNoisyImages_face2.png', cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_2: ', meanSquaredError(image_face_2, image_face_enhanced_2))
    # print('MAE on image_face_2: ', meanAbsoluteError(image_face_2, image_face_enhanced_2))
    # print('SNR on image_face_2: ', signalToNoiseRatio(image_face_2, image_face_enhanced_2))
    # print("")
    #
    # image_face_3 = cv2.imread('Dataset/GrayscaleNoisyImages/face_3.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_gaussian_noise_effect(image_face_3, 'SavedImages/GrayscaleNoisyImages_face3.png', 'SavedImages/Plot_GrayscaleNoisyImages_face3.png')
    # image_face_enhanced_3 = cv2.imread('SavedImages/GrayscaleNoisyImages_face3.png', cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_3: ', meanSquaredError(image_face_3, image_face_enhanced_3))
    # print('MAE on image_face_3: ', meanAbsoluteError(image_face_3, image_face_enhanced_3))
    # print('SNR on image_face_3: ', signalToNoiseRatio(image_face_3, image_face_enhanced_3))
    # print("")
    #
    # image_face_4 = cv2.imread('Dataset/GrayscaleNoisyImages/face_4.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_gaussian_noise_effect(image_face_4, 'SavedImages/GrayscaleNoisyImages_face4.png', 'SavedImages/Plot_GrayscaleNoisyImages_face4.png')
    # image_face_enhanced_4 = cv2.imread('SavedImages/GrayscaleNoisyImages_face4.png', cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_4: ', meanSquaredError(image_face_4, image_face_enhanced_4))
    # print('MAE on image_face_4: ', meanAbsoluteError(image_face_4, image_face_enhanced_4))
    # print('SNR on image_face_4: ', signalToNoiseRatio(image_face_4, image_face_enhanced_4))
    # print("")
    #
    # image_face_5 = cv2.imread('Dataset/GrayscaleNoisyImages/face_5.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_gaussian_noise_effect(image_face_5, 'SavedImages/GrayscaleNoisyImages_face5.png', 'SavedImages/Plot_GrayscaleNoisyImages_face5.png')
    # image_face_enhanced_5 = cv2.imread('SavedImages/GrayscaleNoisyImages_face5.png', cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_5: ', meanSquaredError(image_face_5, image_face_enhanced_5))
    # print('MAE on image_face_5: ', meanAbsoluteError(image_face_5, image_face_enhanced_5))
    # print('SNR on image_face5: ', signalToNoiseRatio(image_face_5,image_face_enhanced_5))
    # print("")
    #
    # image_face_thermogram = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_salt_pepper_noise(image_face_thermogram, 'SavedImages/GrayscaleNoisyImages_face_thermogram.png','SavedImages/Plot_GrayscaleNoisyImages_face_thermogram.png')
    # image_face_thermogram_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_thermogram.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_thermogram: ', meanSquaredError(image_face_thermogram, image_face_thermogram_enhanced))
    # print('MAE on image_face_thermogram: ', meanAbsoluteError(image_face_thermogram, image_face_thermogram_enhanced))
    # print('SNR on image_face_thermogram: ', signalToNoiseRatio(image_face_thermogram,image_face_thermogram_enhanced))
    # print("")
    #
    # image_face_thermogram_1 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_1.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_salt_pepper_noise(image_face_thermogram_1, 'SavedImages/GrayscaleNoisyImages_face_thermogram1.png','SavedImages/Plot_GrayscaleNoisyImages_face_thermogram1.png')
    # image_face_thermogram1_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_thermogram1.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_thermogram1: ', meanSquaredError(image_face_thermogram_1, image_face_thermogram1_enhanced))
    # print('MAE on image_face_thermogram1: ', meanAbsoluteError(image_face_thermogram_1, image_face_thermogram1_enhanced))
    # print('SNR on image_face_thermogram1: ', signalToNoiseRatio(image_face_thermogram_1,image_face_thermogram1_enhanced))
    # print("")
    #
    # image_face_thermogram_2 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_2.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_salt_pepper_noise(image_face_thermogram_2, 'SavedImages/GrayscaleNoisyImages_face_thermogram2.png','SavedImages/Plot_GrayscaleNoisyImages_face_thermogram2.png')
    # image_face_thermogram2_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_thermogram2.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_thermogram2: ', meanSquaredError(image_face_thermogram_2, image_face_thermogram2_enhanced))
    # print('MAE on image_face_thermogram2: ', meanAbsoluteError(image_face_thermogram_2, image_face_thermogram2_enhanced))
    # print('SNR on image_face_thermogram2: ', signalToNoiseRatio(image_face_thermogram_2,image_face_thermogram2_enhanced))
    # print("")
    #
    # image_face_thermogram_3 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_3.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_salt_pepper_noise(image_face_thermogram_3, 'SavedImages/GrayscaleNoisyImages_face_thermogram3.png','SavedImages/Plot_GrayscaleNoisyImages_face_thermogram3.png')
    # image_face_thermogram3_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_thermogram3.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_thermogram3: ', meanSquaredError(image_face_thermogram_3, image_face_thermogram3_enhanced))
    # print('MAE on image_face_thermogram3: ', meanAbsoluteError(image_face_thermogram_3, image_face_thermogram3_enhanced))
    # print('SNR on image_face_thermogram3: ', signalToNoiseRatio(image_face_thermogram_3,image_face_thermogram3_enhanced))
    # print("")
    #
    # image_face_thermogram_4 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_4.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_salt_pepper_noise(image_face_thermogram_4, 'SavedImages/GrayscaleNoisyImages_face_thermogram4.png','SavedImages/Plot_GrayscaleNoisyImages_face_thermogram4.png')
    # image_face_thermogram4_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_thermogram4.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_thermogram4: ', meanSquaredError(image_face_thermogram_4, image_face_thermogram4_enhanced))
    # print('MAE on image_face_thermogram4: ', meanAbsoluteError(image_face_thermogram_4, image_face_thermogram4_enhanced))
    # print('SNR on image_face_thermogram4: ', signalToNoiseRatio(image_face_thermogram_4,image_face_thermogram4_enhanced))
    # print("")
    #
    # image_face_thermogram_5 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_5.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_salt_pepper_noise(image_face_thermogram_5, 'SavedImages/GrayscaleNoisyImages_face_thermogram5.png','SavedImages/Plot_GrayscaleNoisyImages_face_thermogram5.png')
    # image_face_thermogram5_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_thermogram5.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_thermogram5: ', meanSquaredError(image_face_thermogram_5, image_face_thermogram5_enhanced))
    # print('MAE on image_face_thermogram5: ', meanAbsoluteError(image_face_thermogram_5, image_face_thermogram5_enhanced))
    # print('SNR on image_face_thermogram5: ', signalToNoiseRatio(image_face_thermogram_5,image_face_thermogram5_enhanced))
    # print("")
    #
    # image_face_finger = cv2.imread('Dataset/GrayscaleNoisyImages/finger.bmp', cv2.IMREAD_GRAYSCALE)
    # image_face_finger_1 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_1.bmp', cv2.IMREAD_GRAYSCALE)
    # image_face_finger_2 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_2.bmp', cv2.IMREAD_GRAYSCALE)
    # image_face_finger_3 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_3.bmp', cv2.IMREAD_GRAYSCALE)
    # image_face_finger_4 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_4.bmp', cv2.IMREAD_GRAYSCALE)
    # image_face_finger_5 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_5.bmp', cv2.IMREAD_GRAYSCALE)
    #
    # display_spatial_filter_type(image_face_finger, 'SavedImages/face_finger_low_pass.png','SavedImages/Plot_face_finger_low_pass.png', 'low_pass')
    # display_spatial_filter_type(image_face_finger_1, 'SavedImages/face_finger1_low_pass.png','SavedImages/Plot_face_finger1_low_pass.png', 'low_pass')
    # display_spatial_filter_type(image_face_finger_2, 'SavedImages/face_finger2_low_pass.png','SavedImages/Plot_face_finger2_low_pass.png', 'low_pass')
    # display_spatial_filter_type(image_face_finger_3, 'SavedImages/face_finger3_low_pass.png','SavedImages/Plot_face_finger3_low_pass.png', 'low_pass')
    # display_spatial_filter_type(image_face_finger_4, 'SavedImages/face_finger4_low_pass.png','SavedImages/Plot_face_finger4_low_pass.png', 'low_pass')
    # display_spatial_filter_type(image_face_finger_5, 'SavedImages/face_finger5_low_pass.png','SavedImages/Plot_face_finger5_low_pass.png', 'low_pass')
    #
    # image_face_finger_low_pass = cv2.imread('SavedImages/face_finger_low_pass.png', cv2.IMREAD_GRAYSCALE)
    # image_face_finger1_low_pass = cv2.imread('SavedImages/face_finger1_low_pass.png', cv2.IMREAD_GRAYSCALE)
    # image_face_finger2_low_pass = cv2.imread('SavedImages/face_finger2_low_pass.png', cv2.IMREAD_GRAYSCALE)
    # image_face_finger3_low_pass = cv2.imread('SavedImages/face_finger3_low_pass.png', cv2.IMREAD_GRAYSCALE)
    # image_face_finger4_low_pass = cv2.imread('SavedImages/face_finger4_low_pass.png', cv2.IMREAD_GRAYSCALE)
    # image_face_finger5_low_pass = cv2.imread('SavedImages/face_finger5_low_pass.png', cv2.IMREAD_GRAYSCALE)
    #
    # print('MSE on image_face_finger_low_pass: ',meanSquaredError(image_face_finger, image_face_finger_low_pass))
    # print('MAE on image_face_finger_low_pass: ',meanAbsoluteError(image_face_finger, image_face_finger_low_pass))
    # print('SNR on image_face_finger_low_pass: ', signalToNoiseRatio(image_face_finger, image_face_finger_low_pass))
    # print("")
    #
    # print('MSE on image_face_finger1_low_pass: ',meanSquaredError(image_face_finger_1, image_face_finger1_low_pass))
    # print('MAE on image_face_finger1_low_pass: ', meanAbsoluteError(image_face_finger_1, image_face_finger1_low_pass))
    # print('SNR on image_face_finger1_low_pass: ', signalToNoiseRatio(image_face_finger_1, image_face_finger1_low_pass))
    # print("")
    #
    # print('MSE on image_face_finger2_low_pass: ',meanSquaredError(image_face_finger_2, image_face_finger2_low_pass))
    # print('MAE on image_face_finger2_low_pass: ', meanAbsoluteError(image_face_finger_2, image_face_finger2_low_pass))
    # print('SNR on image_face_finger2_low_pass: ', signalToNoiseRatio(image_face_finger_2, image_face_finger2_low_pass))
    # print("")
    #
    # print('MSE on image_face_finger3_low_pass: ',meanSquaredError(image_face_finger_3, image_face_finger3_low_pass))
    # print('MAE on image_face_finger3_low_pass: ', meanAbsoluteError(image_face_finger_3, image_face_finger3_low_pass))
    # print('SNR on image_face_finger3_low_pass: ', signalToNoiseRatio(image_face_finger_3, image_face_finger3_low_pass))
    # print("")
    #
    # print('MSE on image_face_finger4_low_pass: ', meanSquaredError(image_face_finger_4, image_face_finger4_low_pass))
    # print('MAE on image_face_finger4_low_pass: ', meanAbsoluteError(image_face_finger_4, image_face_finger4_low_pass))
    # print('SNR on image_face_finger4_low_pass: ', signalToNoiseRatio(image_face_finger_4, image_face_finger4_low_pass))
    # print("")
    #
    # print('MSE on image_face_finger5_low_pass: ', meanSquaredError(image_face_finger_5, image_face_finger5_low_pass))
    # print('MAE on image_face_finger5_low_pass: ', meanAbsoluteError(image_face_finger_5, image_face_finger5_low_pass))
    # print('SNR on image_face_finger5_low_pass: ', signalToNoiseRatio(image_face_finger_5, image_face_finger5_low_pass))
    # print("")
    #
    # image_face_iris = cv2.imread('Dataset/GrayscaleNoisyImages/iris.bmp',cv2.IMREAD_GRAYSCALE)
    # display_without_blur_effect(image_face_iris, 'SavedImages/GrayscaleNoisyImages_face_iris.png','SavedImages/Plot_GrayscaleNoisyImages_face_iris.png')
    # image_face_iris_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_iris.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_iris: ', meanSquaredError(image_face_iris, image_face_iris_enhanced))
    # print('MAE on image_face_iris: ', meanAbsoluteError(image_face_iris, image_face_iris_enhanced))
    # print('SNR on image_face_iris: ', signalToNoiseRatio(image_face_iris,image_face_iris_enhanced))
    # print("")
    #
    # image_face_iris_1 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_1.bmp',cv2.IMREAD_GRAYSCALE)
    # display_without_blur_effect(image_face_iris_1, 'SavedImages/GrayscaleNoisyImages_face_iris_1.png','SavedImages/Plot_GrayscaleNoisyImages_face_iris1.png')
    # image_face_iris1_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_iris_1.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_iris1: ', meanSquaredError(image_face_iris_1, image_face_iris1_enhanced))
    # print('MAE on image_face_iris1: ', meanAbsoluteError(image_face_iris_1, image_face_iris1_enhanced))
    # print('SNR on image_face_iris1: ', signalToNoiseRatio(image_face_iris_1,image_face_iris1_enhanced))
    # print("")
    #
    # image_face_iris_2 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_2.bmp',cv2.IMREAD_GRAYSCALE)
    # display_without_blur_effect(image_face_iris_2,'SavedImages/GrayscaleNoisyImages_face_iris_2.png','SavedImages/Plot_GrayscaleNoisyImages_face_iris2.png')
    # image_face_iris_2_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_iris_2.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_iris2: ', meanSquaredError(image_face_iris_2, image_face_iris_2_enhanced))
    # print('MAE on image_face_iris2: ', meanAbsoluteError(image_face_iris_2, image_face_iris_2_enhanced))
    # print('SNR on image_face_iris2: ', signalToNoiseRatio(image_face_iris_2,image_face_iris_2_enhanced))
    # print("")
    #
    # image_face_iris_3 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_3.bmp',cv2.IMREAD_GRAYSCALE)
    # display_without_blur_effect(image_face_iris_3, 'SavedImages/GrayscaleNoisyImages_face_iris_3.png','SavedImages/Plot_GrayscaleNoisyImages_face_iris_3.png')
    # image_face_iris_3_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_iris_3.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_iris_3: ', meanSquaredError(image_face_iris_3, image_face_iris_3_enhanced))
    # print('MAE on image_face_iris_3: ', meanAbsoluteError(image_face_iris_3, image_face_iris_3_enhanced))
    # print('SNR on image_face_iris_3: ', signalToNoiseRatio(image_face_iris_3,image_face_iris_3_enhanced))
    # print("")
    #
    # image_face_iris_4 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_4.bmp',cv2.IMREAD_GRAYSCALE)
    # display_without_blur_effect(image_face_iris_4, 'SavedImages/GrayscaleNoisyImages_face_iris_4.png','SavedImages/Plot_GrayscaleNoisyImages_face_iris_4.png')
    # image_face_iris_4_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_iris_4.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_iris_4: ', meanSquaredError(image_face_iris_4, image_face_iris_4_enhanced))
    # print('MAE on image_face_iris_4: ', meanAbsoluteError(image_face_iris_4, image_face_iris_4_enhanced))
    # print('SNR on image_face_iris_4: ', signalToNoiseRatio(image_face_iris_4,image_face_iris_4_enhanced))
    # print("")
    #
    # image_face_iris_5 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_5.bmp',cv2.IMREAD_GRAYSCALE)
    # display_without_blur_effect(image_face_iris_5, 'SavedImages/GrayscaleNoisyImages_face_iris_5.png','SavedImages/Plot_GrayscaleNoisyImages_face_iris_5.png')
    # image_face_iris_5_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_iris_5.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_iris_5: ', meanSquaredError(image_face_iris_5, image_face_iris_5_enhanced))
    # print('MAE on image_face_iris_5: ', meanAbsoluteError(image_face_iris_5, image_face_iris_5_enhanced))
    # print('SNR on image_face_iris_5: ', signalToNoiseRatio(image_face_iris_5,image_face_iris_5_enhanced))
    # print("")
    #
    # #########################################INIT VARIABLES RGB NOISY PICS#############################################
    #
    # image_face_RGB = cv2.imread('Dataset/RGBNoisyImages/face.png',cv2.COLOR_BGR2RGB)
    #
    # image_face_RGB_1 = cv2.imread('Dataset/RGBNoisyImages/face1.png',cv2.COLOR_BGR2RGB)
    # remove_salt_and_pepper_noise_from_rgb_images(image_face_RGB_1, 'SavedImages/RGB_Noisy_image_face_1.png', 'SavedImages/Plot_RGB_Image_face_1.png')
    # image_face_1_RGB_enhanced = cv2.imread('SavedImages/RGB_Noisy_image_face_1.png',cv2.COLOR_BGR2RGB)
    # print('MSE on image_face_RGB: ', meanSquaredError(image_face_RGB, image_face_1_RGB_enhanced))
    # print('MAE on image_face_RGB: ', meanAbsoluteError(image_face_RGB, image_face_1_RGB_enhanced))
    # print('SNR on image_face_RGB: ', signalToNoiseRatio(image_face_RGB, image_face_1_RGB_enhanced))
    # print("")
    #
    # image_face_RGB_2 = cv2.imread('Dataset/RGBNoisyImages/face2.png',cv2.COLOR_BGR2RGB)
    # apply_color_map_to_image(image_face_RGB_2, 'SavedImages/RGB_Noisy_image_face_2.png', 'SavedImages/Plot_RGB_Image_face_2.png', 'ocean')
    # image_face_2_RGB_enhanced = cv2.imread('SavedImages/RGB_Noisy_image_face_2.png',cv2.COLOR_BGR2RGB)
    # print('MSE on image_face_RGB_2: ', meanSquaredError(image_face_RGB_2, image_face_2_RGB_enhanced))
    # print('MAE on image_face_RGB_2: ', meanAbsoluteError(image_face_RGB_2, image_face_2_RGB_enhanced))
    # print('SNR on image_face_RGB_2: ', signalToNoiseRatio(image_face_RGB_2, image_face_2_RGB_enhanced))
    # print("")
    #
    # image_face_RGB_3 = cv2.imread('Dataset/RGBNoisyImages/face3.png',cv2.COLOR_BGR2RGB)
    # apply_color_map_to_image(image_face_RGB_3, 'SavedImages/RGB_Noisy_image_face_3.png', 'SavedImages/Plot_RGB_Image_face_3.png', 'hsv')
    # image_face_3_RGB_enhanced = cv2.imread('SavedImages/RGB_Noisy_image_face_3.png',cv2.COLOR_BGR2RGB)
    # print('MSE on image_face_RGB_3: ', meanSquaredError(image_face_RGB_3, image_face_3_RGB_enhanced))
    # print('MAE on image_face_RGB_3: ', meanAbsoluteError(image_face_RGB_3, image_face_3_RGB_enhanced))
    # print('SNR on image_face_RGB_3: ', signalToNoiseRatio(image_face_RGB_3, image_face_3_RGB_enhanced))
    # print("")

    # image_face_RGB_4 = cv2.imread('Dataset/RGBNoisyImages/face4.png')
    # remove_red_background_from_thermal_image(image_face_RGB_4, 'SavedImages/RGB_Noisy_image_face_4.png',
    #                                          'SavedImages/Plot_RGB_Image_face_4.png')
    # image_face_4_RGB_enhanced = cv2.imread('SavedImages/RGB_Noisy_image_face_4.png')
    # print('MSE on image_face_RGB_4: ', meanSquaredError(image_face_RGB_4, image_face_4_RGB_enhanced))
    # print('MAE on image_face_RGB_4: ', meanAbsoluteError(image_face_RGB_4, image_face_4_RGB_enhanced))
    # print('SNR on image_face_RGB_4: ', signalToNoiseRatio(image_face_RGB_4, image_face_4_RGB_enhanced))
    # print("")
    #
    # # image_face_lena = cv2.imread('Dataset/RGBNoisyImages/lena.png')
    # #
    # # image_face_lena1 = cv2.imread('Dataset/RGBNoisyImages/lena1.png')
    # # image_face_lena1_RGB = cv2.cvtColor(image_face_lena1, cv2.COLOR_BGR2RGB)
    # #
    # # increase_brightness(image_face_lena1_RGB, 'SavedImages/Lena1_brighness.png', 'SavedImages/Plot_RGB_Image_Lena_1.png')
    # # image_face_lena1_brightness = cv2.imread('SavedImages/Lena1_brighness.png')
    # #
    # # image_face_lena1_brightness_RGB = cv2.cvtColor(image_face_lena1_brightness, cv2.COLOR_BGR2RGB)
    # # print('MSE on image_face_lena1_RGB: ', meanSquaredError(image_face_lena1_RGB, image_face_lena1_brightness))
    # # print('MAE on image_face_lena1_RGB: ', meanAbsoluteError(image_face_lena1_RGB, image_face_lena1_brightness))
    # # print('SNR on image_face_lena1_RGB: ', signalToNoiseRatio(image_face_lena1_RGB, image_face_lena1_brightness))
    # # print("")
    # #
    # # image_face_lena2 = cv2.imread('Dataset/RGBNoisyImages/lena2.png')
    # # image_face_lena2_RGB = cv2.cvtColor(image_face_lena2, cv2.COLOR_BGR2RGB)
    # # increase_contrast(image_face_lena2_RGB, 'SavedImages/Lena2_Contrast.png', 'SavedImages/Plot_RGB_Image_Lena_2.png')
    # #
    # # image_face_lena2_contrast = cv2.imread('SavedImages/Lena2_Contrast.png')
    # # image_face_lena2_contrast_RGB = cv2.cvtColor(image_face_lena, cv2.COLOR_BGR2RGB)
    # #
    # # print('MSE on image_face_lena2_RGB: ', meanSquaredError(image_face_lena2_RGB, image_face_lena2_contrast_RGB))
    # # print('MAE on image_face_lena2_RGB: ', meanAbsoluteError(image_face_lena2_RGB, image_face_lena2_contrast_RGB))
    # # print('SNR on image_face_lena2_RGB: ', signalToNoiseRatio(image_face_lena2_RGB, image_face_lena2_contrast_RGB))
    # # print("")
    # #
    # # image_face_lena3 = cv2.imread('Dataset/RGBNoisyImages/lena3.png')
    # # image_face_lena3_RGB = cv2.cvtColor(image_face_lena3, cv2.COLOR_BGR2RGB)
    # #
    # # increase_brightness(image_face_lena3_RGB, 'SavedImages/Lena3_brighness.png', 'SavedImages/Plot_RGB_Image_Lena_3.png', factor=2.5)
    # # image_face_lena3_brightness_RGB = cv2.imread('SavedImages/Lena3_brighness.png')
    # #
    # # image_face_lena3_brightness_RGB = cv2.cvtColor(image_face_lena3_brightness_RGB, cv2.COLOR_BGR2RGB)
    # # print('MSE on image_face_lena3_RGB: ', meanSquaredError(image_face_lena3_RGB, image_face_lena3_brightness_RGB))
    # # print('MAE on image_face_lena3_RGB: ', meanAbsoluteError(image_face_lena3_RGB, image_face_lena3_brightness_RGB))
    # # print('SNR on image_face_lena3_RGB: ', signalToNoiseRatio(image_face_lena3_RGB, image_face_lena3_brightness_RGB))
    # # print("")
    # #
    image_face_lena4 = cv2.imread('Dataset/RGBNoisyImages/lena4.png')
    image_face_lena4_RGB = cv2.cvtColor(image_face_lena4, cv2.COLOR_BGR2RGB)
    display_without_gaussian_noise_effect(image_face_lena4_RGB, 'SavedImages/RGB_face4_RGB.png', 'SavedImages/Plot_RGB_Lena4_RGB.png')
    image_face_lena4_enhanced = cv2.imread('SavedImages/RGB_face4_RGB.png')
    image_face_lena4_RGB_enhanced = cv2.cvtColor(image_face_lena4_enhanced, cv2.COLOR_BGR2RGB)
    print('MSE on image_face_lena4_RGB: ', meanSquaredError(image_face_lena4_RGB, image_face_lena4_RGB_enhanced))
    print('MAE on image_face_lena4_RGB: ', meanAbsoluteError(image_face_lena4_RGB, image_face_lena4_RGB_enhanced))
    print('SNR on image_face_lena4_RGB: ', signalToNoiseRatio(image_face_lena4_RGB, image_face_lena4_RGB_enhanced))
    print("")

    image_monarch = cv2.imread('Dataset/RGBNoisyImages/monarch.png')

    image_monarch1 = cv2.imread('Dataset/RGBNoisyImages/monarch1.png', cv2.COLOR_BGR2RGB)
    fix_magenta_color_rgb(image_monarch1, 'SavedImages/Monarch1_enhanced.png', 'SavedImages/Plot_RGB_monarch_1_RGB.png')
    image_monarch1_enhanced = cv2.imread('SavedImages/Monarch1_enhanced.png', cv2.COLOR_BGR2RGB)
    print('MSE on image_monarch1: ', meanSquaredError(image_monarch1, image_monarch1_enhanced))
    print('MEA on image_monarch1: ', meanAbsoluteError(image_monarch1, image_monarch1_enhanced))
    print('SNR on image_monarch1: ', signalToNoiseRatio(image_monarch1, image_monarch1_enhanced))
    print("")

    image_monarch2 = cv2.imread('Dataset/RGBNoisyImages/monarch2.png')
    fix_blue_color_rgb(image_monarch2, 'SavedImages/Monarch2_enhanced.png', 'SavedImages/Plot_RGB_monarch_2_RGB.png')
    image_monarch2_enhanced = cv2.imread('SavedImages/Monarch2_enhanced.png')
    print('MSE on image_monarch2: ', meanSquaredError(image_monarch2, image_monarch2_enhanced))
    print('MEA on image_monarch2: ', meanAbsoluteError(image_monarch2, image_monarch2_enhanced))
    print('SNR on image_monarch2: ', signalToNoiseRatio(image_monarch2, image_monarch2_enhanced))
    print("")

    image_monarch3 = cv2.imread('Dataset/RGBNoisyImages/monarch3.png', cv2.COLOR_BGR2RGB)
    remove_blur_and_enhance_contrast(image_monarch3, 'SavedImages/Monarch3_enhanced.png','SavedImages/Plot_RGB_monarch_3_RGB.png')
    image_monarch3_enhanced = cv2.imread('SavedImages/Monarch3_enhanced.png')
    print('MSE on image_monarch3: ', meanSquaredError(image_monarch3, image_monarch3_enhanced))
    print('MEA on image_monarch3: ', meanAbsoluteError(image_monarch3, image_monarch3_enhanced))
    print('SNR on image_monarch3: ', signalToNoiseRatio(image_monarch3, image_monarch3_enhanced))
    print("")

    image_monarch4 = cv2.imread('Dataset/RGBNoisyImages/monarch4.png', cv2.COLOR_BGR2RGB)
    remove_red_noise_rgb_image(image_monarch4, 'SavedImages/Monarch4_enhanced.png','SavedImages/Plot_RGB_monarch_4_RGB.png')
    image_monarch4_enhanced = cv2.imread('SavedImages/Monarch4_enhanced.png', cv2.COLOR_BGR2RGB)
    print('MSE on image_monarch4: ', meanSquaredError(image_monarch3, image_monarch4_enhanced))
    print('MEA on image_monarch4: ', meanAbsoluteError(image_monarch3, image_monarch4_enhanced))
    print('SNR on image_monarch4: ', signalToNoiseRatio(image_monarch3, image_monarch4_enhanced))
    print("")