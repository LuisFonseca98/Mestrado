import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

""""
Applies Brightness to an image
"""
def applyBrightness(pathToSave,image):

    imgRead = cv2.imread(image)
    imgBright = cv2.convertScaleAbs(imgRead, alpha=1.5, beta=8)

    cv2.imwrite(pathToSave, imgBright)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
Applies contrast to the image
"""
def applyContrast(pathToSave,image):

    contrast = 5.  # Contrast control ( 0 to 127)
    brightness = 2.  # Brightness control (0-100)
    imgRead = cv2.imread(image)
    imgContrasted = cv2.addWeighted(imgRead,contrast,image,0,brightness)

    cv2.imwrite(pathToSave, imgContrasted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
Calculates the following values:
->Entropy
->Contrast
->Brightness
"""
def calculateImageValues(image):

    #init variables
    imgRead = cv2.imread(image)
    gray_image = cv2.cvtColor(imgRead, cv2.COLOR_BGR2GRAY)
    width, height = gray_image.shape

    max_value = float("-inf")
    min_value = float("inf")

    #loop through pixel image
    for i in range(width):
        for j in range(height):
            pixel_value = gray_image[i,j]

            #store the max and min pixel value of an image
            max_value = max(max_value,np.max(pixel_value))
            min_value = min(min_value, np.min(pixel_value))


    #calculate the contrast (formula used in the worksheet)
    contrast = 20 * np.log10((max_value + 1) / (min_value + 1))

    #calculate the average intensity (brightness)
    brightness = np.mean(gray_image)

    #calculate the entropy of an image
    _bins = 256
    hist, _ = np.histogram(gray_image.ravel(), bins=_bins, range=(0, _bins))
    prob_dist = hist / hist.sum()
    image_entropy = entropy(prob_dist, base=2)

    print(f"Min value {min_value} | Max value {max_value}")

    print(f"Entropy value {round(image_entropy,2)} | "
          f"Brightness value {round(brightness,2)}| "
          f"Contrast value {round(contrast,2)}")


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

def obtainSpectrumPhase(image):

    imgRead = cv2.imread(image)
    img = cv2.cvtColor(imgRead,cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    phase_spectrum = np.angle(dft_shift)
    return phase_spectrum

def displayPlotImages(title,image):

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
    plt.show()




