import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

""""
Applys the homomorphic filter
"""


def homomorphic_filter(img, gamma_low, gamma_high, cutoff):
    log_img = np.log1p(np.float32(img))
    f_img = np.fft.fft2(log_img)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = cutoff
    center = (crow, ccol)
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0
    f_img = f_img * mask
    img_back = np.fft.ifft2(f_img)
    img_result = np.exp(np.real(img_back)) - 1
    img_result = cv2.normalize(img_result, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(img_result)


"""
Applys the Gaussian Filter on a image
"""


def applyGaussianFilterInImage(image, imageName):
    imgRead = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(imgRead, ksize=(5, 5), sigmaX=0)
    cv2.imwrite('SavedImages/' + imageName + '.png', blurred_image)
    return blurred_image


"""
Applys a Laplacian Gaussian Filter on an image
"""


def applyLaplacianGaussianFilterInImage(image, imageName):
    imgRead = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(imgRead, ksize=(5, 5), sigmaX=0)
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    convertLaplacian = cv2.convertScaleAbs(laplacian)
    cv2.imwrite('SavedImages/' + imageName + '.png', convertLaplacian)
    return convertLaplacian


"""
Applys erosion to an image
"""


def applyErosionOnImage(image):
    imgRead = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(imgRead, kernel, iterations=1)
    cv2.imwrite('SavedImages/Image_Eroded.png', eroded_image)
    return eroded_image


"""
Applys dilation to an image
"""


def applyDilationOnImage(image):
    imgRead = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.dilate(imgRead, kernel, iterations=1)
    cv2.imwrite('SavedImages/Image_Dilated.png', eroded_image)
    return eroded_image


"""
Applys the opening to an image
"""


def applyOpeningOnImage(image):
    imgRead = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.morphologyEx(imgRead, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('SavedImages/Image_Opening.png', eroded_image)
    return eroded_image


"""
Applys the closing technique to an image
"""


def applyClosingOnImage(image):
    imgRead = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.morphologyEx(imgRead, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('SavedImages/Image_Closing.png', eroded_image)
    return eroded_image


""""
Applies Brightness to an image
"""


def applyBrightness(image):
    imgRead = cv2.imread(image)
    imgBright = cv2.convertScaleAbs(imgRead, alpha=1.5, beta=8)
    cv2.imwrite('SavedImages/Iris_Image_Enhancement.png', imgBright)
    return imgBright


"""
Displays the original and enhanced one side by side
"""


def displayBrightnessImage(image, pathToSavePlot):
    imgRead = cv2.imread(image)
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(imgRead, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(applyBrightness(image), cmap='gray')
    plt.title('Enhanced Brightness Image Module')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(pathToSavePlot)
    plt.show()


"""
Calculates the following values:
->Entropy
->Contrast
->Brightness
"""


def calculateImageValues(image):
    # init variables
    imgRead = cv2.imread(image)
    gray_image = cv2.cvtColor(imgRead, cv2.COLOR_BGR2GRAY)
    width, height = gray_image.shape

    max_value = float("-inf")
    min_value = float("inf")

    # loop through pixel image
    for i in range(width):
        for j in range(height):
            pixel_value = gray_image[i, j]

            # store the max and min pixel value of an image
            max_value = max(max_value, np.max(pixel_value))
            min_value = min(min_value, np.min(pixel_value))

    # calculate the contrast (formula used in the worksheet)
    contrast = 20 * np.log10((max_value + 1) / (min_value + 1))

    # calculate the average intensity (brightness)
    brightness = np.mean(gray_image)

    # calculate the entropy of an image
    _bins = 256
    hist, _ = np.histogram(gray_image.ravel(), bins=_bins, range=(0, _bins))
    prob_dist = hist / hist.sum()
    image_entropy = entropy(prob_dist, base=2)

    print(f"Min value {min_value} | Max value {max_value}")

    print(f"Entropy value {round(image_entropy, 2)} | "
          f"Brightness value {round(brightness, 2)}| "
          f"Contrast value {round(contrast, 2)}")


"""
Calculates the spectrum module of an image
"""


def obtainSpectrumModule(image):
    imgRead = cv2.imread(image)
    gray = cv2.cvtColor(imgRead, cv2.COLOR_BGR2GRAY)

    # Compute the discrete Fourier Transform of the image
    fourier = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Shift the zero-frequency component to the center of the spectrum
    fourier_shift = np.fft.fftshift(fourier)

    # calculate the magnitude of the Fourier Transform
    magnitude = 20 * np.log(cv2.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1]))

    # Scale the magnitude for display
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    return magnitude


"""
Displat the phase spectrum of an image
"""


def obtainSpectrumPhase(image):
    imgRead = cv2.imread(image)
    img = cv2.cvtColor(imgRead, cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    phase_spectrum = np.angle(dft_shift)
    return phase_spectrum


"""
Displays the original image, spectrum and phase spectrum of the image, as well as the histogram
"""


def displayPlotImages(title, image, titlePlot):
    # call methods
    imgRead = cv2.imread(image)
    imgSpectrumModule = obtainSpectrumModule(image)
    imgSpectrumPhase = obtainSpectrumPhase(image)
    calculateImageValues(image)

    # Calculate histogram for one of the images
    hist_data = cv2.calcHist([imgRead], [0], None, [256], [0, 256])

    # Normalize the histogram
    hist_data /= hist_data.sum()

    # display  images in a plot (2x2)
    plt.subplot(2, 2, 1)
    plt.imshow(imgRead, cmap='gray')
    plt.title(title)
    plt.axis('off')

    ax = plt.subplot(2, 2, 2)
    ax.set_title('Histogram Image')
    # Add histogram to the twin Axes
    ax_hist = ax.twinx()
    ax_hist.plot(hist_data, color='blue')
    ax_hist.set_yticks([])

    plt.subplot(2, 2, 3)
    plt.imshow(imgSpectrumModule, cmap='gray')
    plt.title('Spectrum Module')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(imgSpectrumPhase, cmap='gray')
    plt.title('Spectrum Phase')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(titlePlot)
    plt.show()




"""
Display the original and enhanced image with erosion
"""


def displayErosionImage(image, pathToSavePlot):
    imgRead = cv2.imread(image)
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(imgRead, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(applyErosionOnImage(image), cmap='gray')
    plt.title('Enhanced Erosion Image Module')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(pathToSavePlot)
    plt.show()


"""
Display the original and enhanced image with dilation
"""


def displayDilatedImage(image, pathToSavePlot):
    imgRead = cv2.imread(image)
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(imgRead, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(applyDilationOnImage(image), cmap='gray')
    plt.title('Enhanced Dilation Image Module')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(pathToSavePlot)
    plt.show()


"""
Display the original and enhanced image with opening
"""


def displayOpeningImage(image, pathToSavePlot):
    imgRead = cv2.imread(image)
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(imgRead, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(applyOpeningOnImage(image), cmap='gray')
    plt.title('Enhanced Opening Image Module')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(pathToSavePlot)
    plt.show()


"""
Display the original and enhanced image with closing
"""


def displayClosingImage(image, pathToSavePlot):
    imgRead = cv2.imread(image)
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(imgRead, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(applyClosingOnImage(image), cmap='gray')
    plt.title('Enhanced Closing Image Module')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(pathToSavePlot)
    plt.show()


def displayGaussingImage(image, imageName, pathToSavePlot):
    imgRead = cv2.imread(image)
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(imgRead, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(applyGaussianFilterInImage(image, imageName), cmap='gray')
    plt.title('Enhanced Gaussing Image Module')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(pathToSavePlot)
    plt.show()


def displayLaplacianGaussingImage(image, imageName, pathToSavePlot):
    imgRead = cv2.imread(image)
    # display  images in a plot (2x2)
    plt.subplot(1, 2, 1)
    plt.imshow(imgRead, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(applyLaplacianGaussianFilterInImage(image, imageName), cmap='gray')
    plt.title('Enhanced Laplacian Gaussing Image Module')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(pathToSavePlot)
    plt.show()


def displayHomomorphicFilterImage(original_image, enhanced_image, pathToSavePlot):
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray'),
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_image, cmap='gray'),
    plt.title('Homomorphic Filter Image')

    plt.tight_layout()
    plt.savefig(pathToSavePlot)
    plt.show()


if __name__ == "__main__":
    ###########################################INIT VARIABLES#############################################
    imageCT = 'Dataset/BiometricMedicalGrayscaleImages/CT.jpg'
    imageFaceThermogram = 'Dataset/BiometricMedicalGrayscaleImages/face_thermogram.png'
    imageFinger = 'Dataset/BiometricMedicalGrayscaleImages/finger.png'
    imageIris = 'Dataset/BiometricMedicalGrayscaleImages/iris.png'
    imageMR = 'Dataset/BiometricMedicalGrayscaleImages/MR.jpg'
    imagePET = 'Dataset/BiometricMedicalGrayscaleImages/PET.png'
    imageThyroid = 'Dataset/BiometricMedicalGrayscaleImages/Thyroid.tif'
    imageXRay = 'Dataset/BiometricMedicalGrayscaleImages/XRay.png'

    # # ##################################EXERCISE 1 ALINEA A)###########################################
    # #display original images without DIP
    # print('ImageCT', imageCT)
    # displayPlotImages('Original Image',imageCT,'SavedImages/Plot ImageCT.png')
    # print("")
    # print('imageFaceThermogram', imageFaceThermogram)
    # displayPlotImages('Original Image',imageFaceThermogram, 'SavedImages/Plot Face Thermogram.png')
    # print("")
    # print('imageFinger', imageFinger)
    # displayPlotImages('Original Image',imageFinger, 'SavedImages/Plot Finger.png')
    # print("")
    # print('imageIris', imageIris)
    # displayPlotImages('Original Image',imageIris, 'SavedImages/Plot Iris.png')
    # print("")
    # print('imageMR', imageMR)
    # displayPlotImages('Original Image',imageMR, 'SavedImages/Plot MR.png')
    # print("")
    # print('imagePET', imagePET)
    # displayPlotImages('Original Image',imagePET, 'SavedImages/Plot PET.png')
    # print("")
    # print('imageThyroid', imageThyroid)
    # displayPlotImages('Original Image',imageThyroid, 'SavedImages/Plot Thyroid.png')
    # print("")
    # print('imageFinger', imageFinger)
    # displayPlotImages('Original Image',imageXRay, 'SavedImages/Plot Finger.png')
    # print("")

    #################################EXERCISE 1 ALINEA B)###########################################
    #####################################CT IMAGE APPLY DIFFERENT FILTERS##########################################################
    # 1) Gaussian filter image
    print("GAUSSIAN FILTER FOR THE CT IMAGE")
    displayGaussingImage(imageCT, 'CT_Gaussian', 'SavedImages/CT_Plot_Image.png')
    image_Gaussing = 'SavedImages/CT_Gaussian.png'
    calculateImageValues(image_Gaussing)
    print("")

    # # 2) Laplacian Gaussian filter image
    # print("LAPLACIAN GAUSSIAN FILTER FOR THE CT IMAGE")
    # displayLaplacianGaussingImage(imageCT, 'CT_Laplacian_Gaussian', 'SavedImages/CT_Laplacian_Gaussing_Plot_Image.png')
    # image_laplacian_Gaussing = 'SavedImages/CT_Laplacian_Gaussian.png'
    # calculateImageValues(image_laplacian_Gaussing)
    # print("")
    #
    # print("HOMOMORPHIC FILTER FOR THE CT IMAGE")
    # # 3) HOMOMORPHIC FILTER
    # original_image = cv2.imread(imageCT, cv2.IMREAD_GRAYSCALE)
    # enhanced_image = homomorphic_filter(original_image, gamma_low=0.3, gamma_high=1.5, cutoff=30)
    # cv2.imwrite('SavedImages/CT_Homomorphic_filter_Image.png', enhanced_image)
    # image_CT_Homomorphic = 'SavedImages/CT_Homomorphic_filter_Image.png'
    # displayHomomorphicFilterImage(original_image,enhanced_image, 'SavedImages/CT_Homomorphic_Plot_Image.png')
    # calculateImageValues(image_CT_Homomorphic)
    # print("")

    # #######################################FACE THERMOGRAM IMAGE APPLY DIFFERENT FILTERS##########################################################
    #1) Gaussian filter image
    # print("GAUSSIAN FILTER FOR THE FACE THERMOGRAM IMAGE")
    # displayGaussingImage(imageFaceThermogram, 'Face_Thermogram_Gaussian', 'SavedImages/Face_Thermogram_Plot_Image.png')
    # image_Gaussing = 'SavedImages/Face_Thermogram_Gaussian.png'
    # calculateImageValues(image_Gaussing)
    # print("")
    #
    # print("LAPLACIAN GAUSSIAN FILTER FOR THE FACE THERMOGRAM IMAGE")
    # # 2) Laplacian Gaussian filter image
    # displayLaplacianGaussingImage(imageFaceThermogram, 'Face_Thermogram_Gaussian_Laplacian_Gaussian', 'SavedImages/Face_Thermogram_Laplacian_Gaussing_Plot_Image.png')
    # image_laplacian_Gaussing = 'SavedImages/Face_Thermogram_Gaussian_Laplacian_Gaussian.png'
    # calculateImageValues(image_laplacian_Gaussing)
    # print("")

    print("HOMOMORPHIC FILTER FOR THE FACE THERMOGRAM IMAGE")
    # 3) HOMOMORPHIC FILTER
    original_image = cv2.imread(imageFaceThermogram, cv2.IMREAD_GRAYSCALE)
    enhanced_image = homomorphic_filter(original_image, gamma_low=0.3, gamma_high=1.5, cutoff=30)
    cv2.imwrite('SavedImages/Face_Thermogram_Homomorphic_filter_Image.png', enhanced_image)
    image_CT_Homomorphic = 'SavedImages/Face_Thermogram_Homomorphic_filter_Image.png'
    displayHomomorphicFilterImage(original_image,enhanced_image, 'SavedImages/Face_Thermogram_Homomorphic_Plot_Image.png')
    calculateImageValues(image_CT_Homomorphic)
    print("")
    #
    # #######################################FINGER IMAGE APPLY DIFFERENT FILTERS##########################################################
    # #EROSION IMAGE
    # displayErosionImage(imageFinger,  'SavedImages/Erosion_Plot_Image.png')
    # finger_eroded = 'SavedImages/Image_Eroded.png'
    # calculateImageValues(finger_eroded)
    #
    print("")
    #DILATION image
    displayDilatedImage(imageFinger,  'SavedImages/Dilatation_Plot_Image.png')
    finger_dilated = 'SavedImages/Image_Dilated.png'
    calculateImageValues(finger_dilated)

    # print("")
    # #OPENING image
    # displayOpeningImage(imageFinger,  'SavedImages/Opening_Plot_Image.png')
    # finger_opening = 'SavedImages/Image_Opening.png'
    # calculateImageValues(finger_opening)
    #
    # print("")
    # #CLOSING image
    # displayClosingImage(imageFinger,  'SavedImages/Closing_Plot_Image.png')
    # finger_closing = 'SavedImages/Image_Closing.png'
    # calculateImageValues(finger_closing)
    #
    # print("")
    # ######################################IRIS BRITGHNESS ENHANCEMENT##########################################################
    print('Brightness Image imageIris')
    displayBrightnessImage(imageIris, 'SavedImages/Brightness_Iris_Plot_Image.png')
    iris_brightness = "SavedImages/Iris_Image_Enhancement.png"
    calculateImageValues(iris_brightness)
    #
    # #######################################MR IMAGE APPLY DIFFERENT FILTERS##########################################################
    # # 1) Gaussian filter image
    # print("GAUSSIAN FILTER FOR THE FACE MR IMAGE")
    # displayGaussingImage(imageMR, 'MR_Gaussian', 'SavedImages/MR_Plot_Image.png')
    # image_Gaussing = 'SavedImages/MR_Gaussian.png'
    # calculateImageValues(image_Gaussing)
    # print("")
    #
    # print("")
    # print("LAPLACIAN GAUSSIAN FILTER FOR THE FACE MR IMAGE")
    # # 2) Laplacian Gaussian filter image
    # displayLaplacianGaussingImage(imageMR, 'MR_Laplacian_Gaussian','SavedImages/MR_Laplacian_Gaussing_Plot_Image.png')
    # image_laplacian_Gaussing = 'SavedImages/MR_Laplacian_Gaussian.png'
    # calculateImageValues(image_laplacian_Gaussing)
    #
    print("")
    print("HOMOMORPHIC FILTER FOR THE MR IMAGE")
    # 3) HOMOMORPHIC FILTER
    original_image = cv2.imread(imageMR, cv2.IMREAD_GRAYSCALE)
    enhanced_image = homomorphic_filter(original_image, gamma_low=0.3, gamma_high=1.5, cutoff=30)
    cv2.imwrite('SavedImages/MR_Homomorphic_filter_Image.png', enhanced_image)
    image_MR_Homomorphic = 'SavedImages/MR_Homomorphic_filter_Image.png'
    displayHomomorphicFilterImage(original_image,enhanced_image, 'SavedImages/MR_Homomorphic_Plot_Image.png')
    calculateImageValues(image_MR_Homomorphic)
    print("")

    # #######################################PET APPLY DIFFERENT FILTERS##########################################################
    # # 1) Gaussian filter image
    # print(" GAUSSIAN FILTER FOR THE PET IMAGE")
    # displayGaussingImage(imagePET, 'PET_Gaussian', 'SavedImages/PET_Plot_Image.png')
    # image_Gaussing = 'SavedImages/PET_Gaussian.png'
    # calculateImageValues(image_Gaussing)
    # print("")
    #
    # # 2) Laplacian Gaussian filter image
    # print("LAPLACIAN GAUSSIAN FILTER FOR THE PET IMAGE")
    # displayLaplacianGaussingImage(imagePET, 'PET_Laplacian_Gaussian','SavedImages/PET_Laplacian_Gaussing_Plot_Image.png')
    # image_laplacian_Gaussing = 'SavedImages/PET_Laplacian_Gaussian.png'
    # calculateImageValues(image_laplacian_Gaussing)
    # print("")
    #
    # 3) HOMOMORPHIC FILTER
    print("HOMOMORPHIC FILTER FOR THE PET IMAGE")
    original_image = cv2.imread(imagePET, cv2.IMREAD_GRAYSCALE)
    enhanced_image = homomorphic_filter(original_image, gamma_low=0.3, gamma_high=1.5, cutoff=30)
    cv2.imwrite('SavedImages/PET_Homomorphic_filter_Image.png', enhanced_image)
    displayHomomorphicFilterImage(original_image, enhanced_image,'SavedImages/PET_Homomorphic_Plot_Image.png')
    image_PET_Homomorphic = 'SavedImages/PET_Homomorphic_filter_Image.png'
    calculateImageValues(image_PET_Homomorphic)
    print("")

    # ######################################THYROID APPLY DIFFERENT FILTERS##########################################################
    # # 1) Gaussian filter image
    # print("GAUSSIAN FILTER FOR THE THYROID IMAGE")
    # displayGaussingImage(imageThyroid, 'Thyroid_Gaussian', 'SavedImages/Thyroid_Plot_Image.png')
    # image_Gaussing = 'SavedImages/Thyroid_Gaussian.png'
    # calculateImageValues(image_Gaussing)
    # print("")
    #
    # 2) Laplacian Gaussian filter image
    print("LAPLACIAN GAUSSIAN FILTER FOR THE THYROID IMAGE")
    displayLaplacianGaussingImage(imageThyroid, 'Thyroid_Laplacian_Gaussian','SavedImages/Thyroid_Laplacian_Gaussing_Plot_Image.png')
    image_laplacian_Gaussing = 'SavedImages/Thyroid_Laplacian_Gaussian.png'
    calculateImageValues(image_laplacian_Gaussing)
    print("")
    #
    # # 3) HOMOMORPHIC FILTER
    # print("HOMOMORPHIC FILTER FOR THE THYROID IMAGE")
    # original_image = cv2.imread(imageThyroid, cv2.IMREAD_GRAYSCALE)
    # enhanced_image = homomorphic_filter(original_image, gamma_low=0.3, gamma_high=1.5, cutoff=30)
    # cv2.imwrite('SavedImages/Thyroid_Homomorphic_filter_Image.png', enhanced_image)
    # image_thyroid_Homomorphic = 'SavedImages/Thyroid_Homomorphic_filter_Image.png'
    # calculateImageValues(image_thyroid_Homomorphic)
    # displayHomomorphicFilterImage(original_image, enhanced_image,'SavedImages/Thyroid_Homomorphic_Plot_Image.png')
    # print("")
    #
    #
    # # #######################################X RAY APPLY DIFFERENT FILTERS##########################################################
    # 1) Gaussian filter image
    print("GAUSSIAN FILTER FOR THE XRAY IMAGE")
    displayGaussingImage(imageXRay, 'XRay_Gaussian', 'SavedImages/XRay_Plot_Image.png')
    image_Gaussing = 'SavedImages/XRay_Gaussian.png'
    calculateImageValues(image_Gaussing)
    print("")

    # # 2) Laplacian Gaussian filter image
    # print("GAUSSIAN LAPLACIAN FILTER FOR THE XRAY IMAGE")
    # displayLaplacianGaussingImage(imageXRay, 'XRay_Laplacian_Gaussian','SavedImages/XRay_Laplacian_Gaussing_Plot_Image.png')
    # image_laplacian_Gaussing = 'SavedImages/XRay_Laplacian_Gaussian.png'
    # calculateImageValues(image_laplacian_Gaussing)
    # print("")
    #
    # print("HOMOMORPHIC FILTER FOR THE XRAY IMAGE")
    # # 3) HOMOMORPHIC FILTER
    # original_image = cv2.imread(imageXRay, cv2.IMREAD_GRAYSCALE)
    # enhanced_image = homomorphic_filter(original_image, gamma_low=0.3, gamma_high=1.5, cutoff=30)
    # cv2.imwrite('SavedImages/XRay_Homomorphic_filter_Image.png', enhanced_image)
    # image_XRay_Homomorphic = 'SavedImages/XRay_Homomorphic_filter_Image.png'
    # calculateImageValues(image_XRay_Homomorphic)
    # displayHomomorphicFilterImage(original_image, enhanced_image,'SavedImages/XRay_Homomorphic_Plot_Image.png')
    # print("")

