import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

"""
Converts an grayscale image to a color one using the 'colormap' function
1. COLORMAP_AUTUMN: Shades of red and yellow.
2. COLORMAP_BONE: Shades of gray, with a light blue tint.
3. COLORMAP_JET: The famous Jet colormap, with blue-cyan-green-yellow-red tones.
4. COLORMAP_WINTER: Shades of blue and green.
5. COLORMAP_RAINBOW: A rainbow spectrum from violet to red.
6. COLORMAP_OCEAN: Shades of blue and green.
7. COLORMAP_SUMMER: Shades of green and yellow.
8. COLORMAP_SPRING: Shades of magenta and yellow.
9. COLORMAP_COOL: Shades of cyan and magenta.
10. COLORMAP_HSV: A colormap representing values in the HSV color space.
"""


def convert_gray_to_pseudocolor_image(image_path, output_path, color_map_type):
    global pseudocolor_image

    if color_map_type == 'jet':
        pseudocolor_image = cv2.applyColorMap(image_path, cv2.COLORMAP_JET)

    elif color_map_type == 'HSV':
        pseudocolor_image = cv2.applyColorMap(image_path, cv2.COLORMAP_HSV)

    elif color_map_type == 'bone':
        pseudocolor_image = cv2.applyColorMap(image_path, cv2.COLORMAP_BONE)

    cv2.imwrite(output_path, pseudocolor_image)
    return cv2.cvtColor(pseudocolor_image, cv2.COLOR_BGR2RGB)


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
    imgRead = cv2.imread(image,cv2.IMREAD_GRAYSCALE)
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
    imgRead = cv2.imread(image,cv2.COLOR_BGR2RGB)
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
    imgRead = cv2.imread(image,cv2.COLOR_BGR2RGB)
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

    imgRead = cv2.imread(image,cv2.CAP_OPENNI_GRAY_IMAGE)
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
    imgRead = cv2.imread(image,cv2.COLOR_BGR2RGB)
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
    plt.imshow(cv2.cvtColor(enhanced_image,cv2.COLOR_BGR2RGB))
    plt.title('Homomorphic Filter Image')

    plt.tight_layout()
    plt.savefig(pathToSavePlot)
    plt.show()


def display_colored_image(original_image, output_path,color_map_type,pathToSavePlot):
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray'),
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(convert_gray_to_pseudocolor_image(original_image,output_path,color_map_type)),
    plt.title('Pseudo color Filter Image')

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
    #####################################CT IMAGE##########################################################
    print("GAUSSIAN FILTER FOR THE CT IMAGE")
    displayGaussingImage(imageCT, 'CT_Gaussian', 'SavedImages/Plot_CT_Image.png')
    image_Gaussing = 'SavedImages/CT_Gaussian.png'
    calculateImageValues(image_Gaussing)
    print("")

    #######################################FINGER IMAGE##########################################################

    print("DILATION FILTER FOR THE FINGER IMAGE")
    # DILATION image
    displayDilatedImage(imageFinger, 'SavedImages/Plot_Dilatation_Image.png')
    finger_dilated = 'SavedImages/Image_Dilated.png'
    calculateImageValues(finger_dilated)
    print("")

    # #######################################THYROID IMAGE##########################################################
    print("EROSION FILTER FOR THE THYROID IMAGE")
    displayErosionImage(imageThyroid,  'SavedImages/Plot_Erosion_Image.png')
    finger_eroded = 'SavedImages/Image_Eroded.png'
    calculateImageValues(finger_eroded)
    print("")

    #####################################IRIS BRITGHNESS ENHANCEMENT##########################################################
    print('Brightness Image imageIris')
    displayBrightnessImage(imageIris, 'SavedImages/Plot_Brightness_Iris_Image.png')
    iris_brightness = "SavedImages/Iris_Image_Enhancement.png"
    calculateImageValues(iris_brightness)

    # #######################################MR IMAGE##########################################################
    print("COLORED FILTER FOR THE MR IMAGE")
    original_image = cv2.imread(imageMR)
    display_colored_image(original_image,'SavedImages/MR_Colored_Image.png','jet','SavedImages/Plot_MR_Pseudocolor_Image.png')
    image_MR_Colored = 'SavedImages/MR_Colored_Image.png'
    calculateImageValues(image_MR_Colored)
    print("")

    ######################################PET##########################################################
    print("HOMOMORPHIC FILTER FOR THE PET IMAGE")
    original_image = cv2.imread(imagePET, cv2.IMREAD_GRAYSCALE)
    enhanced_image = homomorphic_filter(original_image, gamma_low=0.3, gamma_high=1.5, cutoff=30)
    cv2.imwrite('SavedImages/PET_Homomorphic_filter_Image.png', enhanced_image)
    displayHomomorphicFilterImage(original_image, enhanced_image, 'SavedImages/Plot_PET_Homomorphic_Image.png')
    image_PET_Homomorphic = 'SavedImages/PET_Homomorphic_filter_Image.png'
    calculateImageValues(image_PET_Homomorphic)
    print("")

    #######################################FACE THERMOGRAM##########################################################
    print("COLORED FILTER FOR THE FACE THERMOGRAM IMAGE")
    original_image = cv2.imread(imageFaceThermogram)
    display_colored_image(original_image,'SavedImages/Face_Thermogram_Colored_Image.png','jet','SavedImages/Plot_Face_Thermogram_Pseudocolor_Image.png')
    image_Face_Thermogram_Colored = 'SavedImages/Face_Thermogram_Colored_Image.png'
    calculateImageValues(image_Face_Thermogram_Colored)
    print("")

    ######################################X RAY APPLY##########################################################
    original_image = cv2.imread(imageXRay)
    print("COLORED FILTER FOR THE XRAY IMAGE")
    display_colored_image(original_image, 'SavedImages/XRay_Colored.png', 'bone','SavedImages/Plot_XRay_Image.png')
    image_xray_colored = 'SavedImages/XRay_Colored.png'
    calculateImageValues(image_xray_colored)
    print("")

    ##################BACKUP IN CASE ITS NEEDED IN THE FUTURE
    # print("")
    # #OPENING image
    # displayOpeningImage(imageThyroid,  'SavedImages/Opening_Plot_Image.png')
    # finger_opening = 'SavedImages/Image_Opening.png'
    # calculateImageValues(finger_opening)
    #
    # print("")
    # #CLOSING image
    # displayClosingImage(imageThyroid,  'SavedImages/Closing_Plot_Image.png')
    # finger_closing = 'SavedImages/Image_Closing.png'
    # calculateImageValues(finger_closing)


    # # 2) Laplacian Gaussian filter image
    # print("GAUSSIAN LAPLACIAN FILTER FOR THE XRAY IMAGE")
    # displayLaplacianGaussingImage(imageXRay, 'XRay_Laplacian_Gaussian','SavedImages/XRay_Laplacian_Gaussing_Plot_Image.png')
    # image_laplacian_Gaussing = 'SavedImages/XRay_Laplacian_Gaussian.png'
    # calculateImageValues(image_laplacian_Gaussing)
    # print("")
