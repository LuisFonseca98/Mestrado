import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_snr(image_path1, image_path2, signal_roi=(50, 150, 50, 150), noise_roi=(0, 50, 0, 50)):
    # Extract signal and noise regions
    signal_image1 = image_path1[signal_roi[0]:signal_roi[1], signal_roi[2]:signal_roi[3]]
    noise_image1 = image_path1[noise_roi[0]:noise_roi[1], noise_roi[2]:noise_roi[3]]

    # Extract signal and noise regions
    signal_image2 = image_path2[signal_roi[0]:signal_roi[1], signal_roi[2]:signal_roi[3]]
    noise_image2 = image_path2[noise_roi[0]:noise_roi[1], noise_roi[2]:noise_roi[3]]

    # Calculate signal power and noise power
    signal_power_image1 = np.mean(signal_image1 ** 2)
    noise_power_image1 = np.mean(noise_image1 ** 2)

    signal_power_image2 = np.mean(signal_image2 ** 2)
    noise_power_image2 = np.mean(noise_image2 ** 2)

    # Calculate SNR in decibels (dB)
    snr_db_image1 = 10 * np.log10(signal_power_image1 / noise_power_image1)
    snr_db_image2 = 10 * np.log10(signal_power_image2 / noise_power_image2)

    return f'SNR_Image_Original: {np.round(snr_db_image1, 4)}, SNR_Image_Enhanced: {np.round(snr_db_image2, 4)}'


"""
Calculates the MSE between 2 images
"""


def calculate_mean_squared_error(image1, image2):
    if image1.shape != image2.shape:
        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    m, n = image1.shape
    mse = np.mean((image1 - image2) ** 2) / (m * n)

    return np.round(mse, 6)


"""
Calculates the MAE between 2 images
"""


def calculate_mean_absolute_error(image1, image2):
    if image1.shape != image2.shape:
        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

    m, n = image1.shape
    mae = np.mean(np.abs(image1 - image2)) / (m * n)
    return np.round(mae, 6)


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


def display_spatial_filter_type(image, output_name,path_to_save, filter_type):
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
    ###########################################INIT VARIABLES GRAYSCALE PICS#############################################
    # image_face = cv2.imread('Dataset/GrayscaleNoisyImages/face.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_gaussian_noise_effect(image_face, 'SavedImages/GrayscaleNoisyImages_face.png', 'SavedImages/GrayscaleNoisyImages_Plot_face.png')
    # image_face_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face.png', cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face: ', calculate_mean_squared_error(image_face, image_face_enhanced))
    # print('MAE on image_face: ', calculate_mean_absolute_error(image_face, image_face_enhanced))
    # print('SNR on image_face: ', calculate_snr(image_face, image_face_enhanced))
    # print("")
    #
    # image_face_1 = cv2.imread('Dataset/GrayscaleNoisyImages/face_1.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_gaussian_noise_effect(image_face_1, 'SavedImages/GrayscaleNoisyImages_face1.png', 'SavedImages/GrayscaleNoisyImages_Plot_face1.png')
    # image_face_enhanced_1 = cv2.imread('SavedImages/GrayscaleNoisyImages_face1.png', cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_1: ', calculate_mean_squared_error(image_face_1, image_face_enhanced_1))
    # print('MAE on image_face_1: ', calculate_mean_absolute_error(image_face_1, image_face_enhanced_1))
    # print('SNR on image_face_1: ', calculate_snr(image_face_1, image_face_enhanced_1))
    # print("")
    #
    # image_face_2 = cv2.imread('Dataset/GrayscaleNoisyImages/face_2.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_gaussian_noise_effect(image_face_2, 'SavedImages/GrayscaleNoisyImages_face2.png', 'SavedImages/GrayscaleNoisyImages_Plot_face2.png')
    # image_face_enhanced_2 = cv2.imread('SavedImages/GrayscaleNoisyImages_face2.png', cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_2: ', calculate_mean_squared_error(image_face_2, image_face_enhanced_2))
    # print('MAE on image_face_2: ', calculate_mean_absolute_error(image_face_2, image_face_enhanced_2))
    # print('SNR on image_face_2: ', calculate_snr(image_face_2, image_face_enhanced_2))
    # print("")
    #
    # image_face_3 = cv2.imread('Dataset/GrayscaleNoisyImages/face_3.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_gaussian_noise_effect(image_face_3, 'SavedImages/GrayscaleNoisyImages_face3.png', 'SavedImages/GrayscaleNoisyImages_Plot_face3.png')
    # image_face_enhanced_3 = cv2.imread('SavedImages/GrayscaleNoisyImages_face3.png', cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_3: ', calculate_mean_squared_error(image_face_3, image_face_enhanced_3))
    # print('MAE on image_face_3: ', calculate_mean_absolute_error(image_face_3, image_face_enhanced_3))
    # print('SNR on image_face_3: ', calculate_snr(image_face_3, image_face_enhanced_3))
    # print("")
    #
    # image_face_4 = cv2.imread('Dataset/GrayscaleNoisyImages/face_4.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_gaussian_noise_effect(image_face_4, 'SavedImages/GrayscaleNoisyImages_face4.png', 'SavedImages/GrayscaleNoisyImages_Plot_face4.png')
    # image_face_enhanced_4 = cv2.imread('SavedImages/GrayscaleNoisyImages_face4.png', cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_4: ', calculate_mean_squared_error(image_face_4, image_face_enhanced_4))
    # print('MAE on image_face_4: ', calculate_mean_absolute_error(image_face_4, image_face_enhanced_4))
    # print('SNR on image_face_4: ', calculate_snr(image_face_4, image_face_enhanced_4))
    # print("")
    #
    # image_face_5 = cv2.imread('Dataset/GrayscaleNoisyImages/face_5.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_gaussian_noise_effect(image_face_5, 'SavedImages/GrayscaleNoisyImages_face5.png', 'SavedImages/GrayscaleNoisyImages_Plot_face5.png')
    # image_face_enhanced_5 = cv2.imread('SavedImages/GrayscaleNoisyImages_face5.png', cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_5: ', calculate_mean_squared_error(image_face_5, image_face_enhanced_5))
    # print('MAE on image_face_5: ', calculate_mean_absolute_error(image_face_5, image_face_enhanced_5))
    # print('SNR on image_face5: ', calculate_snr(image_face_5,image_face_enhanced_5))
    # print("")

    # image_face_thermogram = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_salt_pepper_noise(image_face_thermogram, 'SavedImages/GrayscaleNoisyImages_face_thermogram.png','SavedImages/GrayscaleNoisyImages_Plot_face_thermogram.png')
    # image_face_thermogram_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_thermogram.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_thermogram: ', calculate_mean_squared_error(image_face_thermogram, image_face_thermogram_enhanced))
    # print('MAE on image_face_thermogram: ', calculate_mean_absolute_error(image_face_thermogram, image_face_thermogram_enhanced))
    # print('SNR on image_face_thermogram: ', calculate_snr(image_face_thermogram,image_face_thermogram_enhanced))
    # print("")
    #
    # image_face_thermogram_1 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_1.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_salt_pepper_noise(image_face_thermogram_1, 'SavedImages/GrayscaleNoisyImages_face_thermogram1.png','SavedImages/GrayscaleNoisyImages_Plot_face_thermogram1.png')
    # image_face_thermogram1_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_thermogram1.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_thermogram1: ', calculate_mean_squared_error(image_face_thermogram_1, image_face_thermogram1_enhanced))
    # print('MAE on image_face_thermogram1: ', calculate_mean_absolute_error(image_face_thermogram_1, image_face_thermogram1_enhanced))
    # print('SNR on image_face_thermogram1: ', calculate_snr(image_face_thermogram_1,image_face_thermogram1_enhanced))
    # print("")
    #
    # image_face_thermogram_2 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_2.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_salt_pepper_noise(image_face_thermogram_2, 'SavedImages/GrayscaleNoisyImages_face_thermogram2.png','SavedImages/GrayscaleNoisyImages_Plot_face_thermogram2.png')
    # image_face_thermogram2_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_thermogram2.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_thermogram2: ', calculate_mean_squared_error(image_face_thermogram_2, image_face_thermogram2_enhanced))
    # print('MAE on image_face_thermogram2: ', calculate_mean_absolute_error(image_face_thermogram_2, image_face_thermogram2_enhanced))
    # print('SNR on image_face_thermogram2: ', calculate_snr(image_face_thermogram_2,image_face_thermogram2_enhanced))
    # print("")
    #
    # image_face_thermogram_3 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_3.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_salt_pepper_noise(image_face_thermogram_3, 'SavedImages/GrayscaleNoisyImages_face_thermogram3.png','SavedImages/GrayscaleNoisyImages_Plot_face_thermogram3.png')
    # image_face_thermogram3_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_thermogram3.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_thermogram3: ', calculate_mean_squared_error(image_face_thermogram_3, image_face_thermogram3_enhanced))
    # print('MAE on image_face_thermogram3: ', calculate_mean_absolute_error(image_face_thermogram_3, image_face_thermogram3_enhanced))
    # print('SNR on image_face_thermogram3: ', calculate_snr(image_face_thermogram_3,image_face_thermogram3_enhanced))
    # print("")
    #
    # image_face_thermogram_4 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_4.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_salt_pepper_noise(image_face_thermogram_4, 'SavedImages/GrayscaleNoisyImages_face_thermogram4.png','SavedImages/GrayscaleNoisyImages_Plot_face_thermogram4.png')
    # image_face_thermogram4_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_thermogram4.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_thermogram4: ', calculate_mean_squared_error(image_face_thermogram_4, image_face_thermogram4_enhanced))
    # print('MAE on image_face_thermogram4: ', calculate_mean_absolute_error(image_face_thermogram_4, image_face_thermogram4_enhanced))
    # print('SNR on image_face_thermogram4: ', calculate_snr(image_face_thermogram_4,image_face_thermogram4_enhanced))
    # print("")
    #
    # image_face_thermogram_5 = cv2.imread('Dataset/GrayscaleNoisyImages/face_thermogram_5.bmp', cv2.IMREAD_GRAYSCALE)
    # display_without_salt_pepper_noise(image_face_thermogram_5, 'SavedImages/GrayscaleNoisyImages_face_thermogram5.png','SavedImages/GrayscaleNoisyImages_Plot_face_thermogram5.png')
    # image_face_thermogram5_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_thermogram5.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_thermogram5: ', calculate_mean_squared_error(image_face_thermogram_5, image_face_thermogram5_enhanced))
    # print('MAE on image_face_thermogram5: ', calculate_mean_absolute_error(image_face_thermogram_5, image_face_thermogram5_enhanced))
    # print('SNR on image_face_thermogram5: ', calculate_snr(image_face_thermogram_5,image_face_thermogram5_enhanced))
    # print("")

    image_face_finger = cv2.imread('Dataset/GrayscaleNoisyImages/finger.bmp', cv2.IMREAD_GRAYSCALE)
    display_spatial_filter_type(image_face_finger, 'SavedImages/face_finger_high_pass.png','SavedImages/Plot_face_finger_high_pass.png','high_pass')
    display_spatial_filter_type(image_face_finger, 'SavedImages/face_finger_low_pass.png','SavedImages/Plot_face_finger_low_pass.png','low_pass')
    display_spatial_filter_type(image_face_finger, 'SavedImages/finger_band_pass.png','SavedImages/Plot_face_finger_band_pass.png','band_pass')
    display_spatial_filter_type(image_face_finger, 'SavedImages/face_finger_band_reject.png','SavedImages/Plot_face_finger_band_reject.png','band_reject')

    image_face_finger_high_pass = cv2.imread('SavedImages/face_finger_high_pass.png',cv2.IMREAD_GRAYSCALE)
    image_face_finger_low_pass = cv2.imread('SavedImages/face_finger_low_pass.png',cv2.IMREAD_GRAYSCALE)
    image_face_finger_band_pass = cv2.imread('SavedImages/finger_band_pass.png',cv2.IMREAD_GRAYSCALE)
    image_face_finger_band_reject = cv2.imread('SavedImages/face_finger_band_reject.png',cv2.IMREAD_GRAYSCALE)

    print('MSE on image_face_finger_high_pass: ', calculate_mean_squared_error(image_face_finger, image_face_finger_high_pass))
    print('MAE on image_face_finger_high_pass: ', calculate_mean_absolute_error(image_face_finger, image_face_finger_high_pass))
    print('SNR on image_face_finger_high_pass: ', calculate_snr(image_face_finger,image_face_finger_high_pass))
    print("")

    print('MSE on image_face_finger_low_pass: ', calculate_mean_squared_error(image_face_finger, image_face_finger_low_pass))
    print('MAE on image_face_finger_low_pass: ', calculate_mean_absolute_error(image_face_finger, image_face_finger_low_pass))
    print('SNR on image_face_finger_low_pass: ', calculate_snr(image_face_finger,image_face_finger_low_pass))
    print("")

    print('MSE on image_face_finger_band_pass: ',calculate_mean_squared_error(image_face_finger, image_face_finger_band_pass))
    print('MAE on image_face_finger_band_pass: ',calculate_mean_absolute_error(image_face_finger, image_face_finger_band_pass))
    print('SNR on image_face_finger_band_pass: ', calculate_snr(image_face_finger, image_face_finger_band_pass))
    print("")

    print('MSE on image_face_finger_band_reject: ',calculate_mean_squared_error(image_face_finger, image_face_finger_band_reject))
    print('MAE on image_face_finger_band_reject: ',calculate_mean_absolute_error(image_face_finger, image_face_finger_band_reject))
    print('SNR on image_face_finger_band_reject: ', calculate_snr(image_face_finger, image_face_finger_band_reject))
    print("")


    # image_face_finger_1 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_1.bmp', cv2.IMREAD_GRAYSCALE)
    # display_spatial_filter_type(image_face_finger_1, 'SavedImages/GrayscaleNoisyImages_face_finger_1_high_pass.png','high_pass')
    # display_spatial_filter_type(image_face_finger_1, 'SavedImages/GrayscaleNoisyImages_face_finger_1_low_pass.png','low_pass')
    # display_spatial_filter_type(image_face_finger_1, 'SavedImages/GrayscaleNoisyImages_face_finger_1_band_pass.png','band_pass')
    # display_spatial_filter_type(image_face_finger_1, 'SavedImages/GrayscaleNoisyImages_face_finger_1_band_reject.png','band_reject')
    #
    # image_face_finger_2 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_2.bmp', cv2.IMREAD_GRAYSCALE)
    # display_spatial_filter_type(image_face_finger_2, 'SavedImages/GrayscaleNoisyImages_face_finger_2_high_pass.png','high_pass')
    # display_spatial_filter_type(image_face_finger_2, 'SavedImages/GrayscaleNoisyImages_face_finger_2_low_pass.png','low_pass')
    # display_spatial_filter_type(image_face_finger_2, 'SavedImages/GrayscaleNoisyImages_face_finger_2_band_pass.png','band_pass')
    # display_spatial_filter_type(image_face_finger_2, 'SavedImages/GrayscaleNoisyImages_face_finger_2_band_reject.png','band_reject')
    #
    # image_face_finger_3 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_3.bmp', cv2.IMREAD_GRAYSCALE)
    # display_spatial_filter_type(image_face_finger_3, 'SavedImages/GrayscaleNoisyImages_face_finger_3_high_pass.png','high_pass')
    # display_spatial_filter_type(image_face_finger_3, 'SavedImages/GrayscaleNoisyImages_face_finger_3_low_pass.png', 'low_pass')
    # display_spatial_filter_type(image_face_finger_3, 'SavedImages/GrayscaleNoisyImages_face_finger_3_band_pass.png', 'band_pass')
    # display_spatial_filter_type(image_face_finger_3, 'SavedImages/GrayscaleNoisyImages_face_finger_3_band_reject.png', 'band_reject')
    #
    # image_face_finger_4 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_4.bmp', cv2.IMREAD_GRAYSCALE)
    # display_spatial_filter_type(image_face_finger_4, 'SavedImages/GrayscaleNoisyImages_face_finger_4.png','high_pass')
    # display_spatial_filter_type(image_face_finger_4, 'SavedImages/GrayscaleNoisyImages_face_finger_4.png','low_pass')
    # display_spatial_filter_type(image_face_finger_4, 'SavedImages/GrayscaleNoisyImages_face_finger_4.png', 'band_pass')
    # display_spatial_filter_type(image_face_finger_4, 'SavedImages/GrayscaleNoisyImages_face_finger_4.png','band_reject')
    #
    # image_face_finger_5 = cv2.imread('Dataset/GrayscaleNoisyImages/finger_5.bmp', cv2.IMREAD_GRAYSCALE)
    # display_spatial_filter_type(image_face_finger_5, 'SavedImages/GrayscaleNoisyImages_face_finger_5.png', 'high_pass')
    # display_spatial_filter_type(image_face_finger_5, 'SavedImages/GrayscaleNoisyImages_face_finger_5.png', 'low_pass')
    # display_spatial_filter_type(image_face_finger_5, 'SavedImages/GrayscaleNoisyImages_face_finger_5.png', 'band_pass')
    # display_spatial_filter_type(image_face_finger_5, 'SavedImages/GrayscaleNoisyImages_face_finger_5.png','band_reject')

    # image_face_iris = cv2.imread('Dataset/GrayscaleNoisyImages/iris.bmp',cv2.IMREAD_GRAYSCALE)
    # display_without_blur_effect(image_face_iris, 'SavedImages/GrayscaleNoisyImages_face_iris.png')
    # image_face_iris_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_iris.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_iris: ', calculate_mean_squared_error(image_face_iris, image_face_iris_enhanced))
    # print('MAE on image_face_iris: ', calculate_mean_absolute_error(image_face_iris, image_face_iris_enhanced))
    # print('SNR on image_face_iris: ', calculate_snr(image_face_iris,image_face_iris_enhanced))
    # print("")
    #
    # image_face_iris_1 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_1.bmp',cv2.IMREAD_GRAYSCALE)
    # display_without_blur_effect(image_face_iris_1, 'SavedImages/GrayscaleNoisyImages_face_iris_1.png')
    # image_face_iris1_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_iris_1.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_iris1: ', calculate_mean_squared_error(image_face_iris_1, image_face_iris1_enhanced))
    # print('MAE on image_face_iris1: ', calculate_mean_absolute_error(image_face_iris_1, image_face_iris1_enhanced))
    # print('SNR on image_face_iris1: ', calculate_snr(image_face_iris_1,image_face_iris1_enhanced))
    # print("")
    #
    # image_face_iris_2 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_2.bmp',cv2.IMREAD_GRAYSCALE)
    # display_without_blur_effect(image_face_iris_2, 'SavedImages/GrayscaleNoisyImages_face_iris_2.png')
    # image_face_iris_2_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_iris_2.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_iris2: ', calculate_mean_squared_error(image_face_iris_2, image_face_iris_2_enhanced))
    # print('MAE on image_face_iris2: ', calculate_mean_absolute_error(image_face_iris_2, image_face_iris_2_enhanced))
    # print('SNR on image_face_iris2: ', calculate_snr(image_face_iris_2,image_face_iris_2_enhanced))
    # print("")
    #
    # image_face_iris_3 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_3.bmp',cv2.IMREAD_GRAYSCALE)
    # display_without_blur_effect(image_face_iris_3, 'SavedImages/GrayscaleNoisyImages_face_iris_3.png')
    # image_face_iris_3_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_iris_3.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_iris_3: ', calculate_mean_squared_error(image_face_iris_3, image_face_iris_3_enhanced))
    # print('MAE on image_face_iris_3: ', calculate_mean_absolute_error(image_face_iris_3, image_face_iris_3_enhanced))
    # print('SNR on image_face_iris_3: ', calculate_snr(image_face_iris_3,image_face_iris_3_enhanced))
    # print("")
    #
    # image_face_iris_4 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_4.bmp',cv2.IMREAD_GRAYSCALE)
    # display_without_blur_effect(image_face_iris_4, 'SavedImages/GrayscaleNoisyImages_face_iris_4.png')
    # image_face_iris_4_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_iris_4.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_iris_4: ', calculate_mean_squared_error(image_face_iris_4, image_face_iris_4_enhanced))
    # print('MAE on image_face_iris_4: ', calculate_mean_absolute_error(image_face_iris_4, image_face_iris_4_enhanced))
    # print('SNR on image_face_iris_4: ', calculate_snr(image_face_iris_4,image_face_iris_4_enhanced))
    # print("")
    #
    # image_face_iris_5 = cv2.imread('Dataset/GrayscaleNoisyImages/iris_5.bmp',cv2.IMREAD_GRAYSCALE)
    # display_without_blur_effect(image_face_iris_5, 'SavedImages/GrayscaleNoisyImages_face_iris_5.png')
    # image_face_iris_5_enhanced = cv2.imread('SavedImages/GrayscaleNoisyImages_face_iris_5.png',cv2.IMREAD_GRAYSCALE)
    # print('MSE on image_face_iris_5: ', calculate_mean_squared_error(image_face_iris_5, image_face_iris_5_enhanced))
    # print('MAE on image_face_iris_5: ', calculate_mean_absolute_error(image_face_iris_5, image_face_iris_5_enhanced))
    # print('SNR on image_face_iris_5: ', calculate_snr(image_face_iris_5,image_face_iris_5_enhanced))
    # print("")

    ###########################################INIT VARIABLES RGB NOISY PICS#############################################
    # image_face_RGB = cv2.imread('Dataset/RGBNoisyImages/face.png')
    # display_without_salt_pepper_noise(image_face_RGB, 'SavedImages/RGB_Noisy_image_face.png')
    #
    # image_face_RGB_1 = cv2.imread('Dataset/RGBNoisyImages/face1.png')
    # display_without_salt_pepper_noise(image_face_RGB_1, 'SavedImages/RGB_Noisy_image_face1.png')
    #
    # image_face_RGB_2 = cv2.imread('Dataset/RGBNoisyImages/face2.png')
    # display_without_salt_pepper_noise(image_face_RGB_2, 'SavedImages/RGB_Noisy_image_face2.png')
    #
    # image_face_RGB_3 = cv2.imread('Dataset/RGBNoisyImages/face3.png')
    # display_without_salt_pepper_noise(image_face_RGB_3, 'SavedImages/RGB_Noisy_image_face3.png')
    #
    # image_face_RGB_4 = cv2.imread('Dataset/RGBNoisyImages/face4.png')
    # display_without_salt_pepper_noise(image_face_RGB_4, 'SavedImages/RGB_Noisy_image_face4.png')

    # image_face_lena = cv2.imread('Dataset/RGBNoisyImages/lena.png')
    # image_face_lena1 = cv2.imread('Dataset/RGBNoisyImages/lena1.png')
    # image_face_lena2 = cv2.imread('Dataset/RGBNoisyImages/lena2.png')
    # image_face_lena3 = cv2.imread('Dataset/RGBNoisyImages/lena3.png')
    # image_face_lena4 = cv2.imread('Dataset/RGBNoisyImages/lena4.png')
    #
    # image_face_monarch = cv2.imread('Dataset/RGBNoisyImages/monarch.png')
    # image_face_monarch1 = cv2.imread('Dataset/RGBNoisyImages/monarch1.png')
    # image_face_monarch2 = cv2.imread('Dataset/RGBNoisyImages/monarch2.png')
    # image_face_monarch3 = cv2.imread('Dataset/RGBNoisyImages/monarch3.png')
    # image_face_monarch4 = cv2.imread('Dataset/RGBNoisyImages/monarch4.png')
