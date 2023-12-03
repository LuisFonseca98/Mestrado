import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy


def changeBrightness(image):
    img = cv2.imread(image)
    imgBright = cv2.convertScaleAbs(img, alpha=1.5, beta=8)
    # cv2.imshow('Image Brightness Enhanced', imgBright)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def changeContrast(image):
    img = cv2.imread(image)
    contrast = 5.  # Contrast control ( 0 to 127)
    brightness = 2.  # Brightness control (0-100)
    imgContrasted = cv2.addWeighted(img,contrast,img,0,brightness)
    # cv2.imshow('Image Contrast Enhanced', imgContrasted)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def calculateEntropy(image):

    img = cv2.imread(image)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width, height = gray_image.shape
    max_value = float("-inf")
    min_value = float("inf")

    for i in range(width):
        for j in range(height):
            pixel_value = gray_image[i,j]

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

    # now we will be loading the image and converting it to grayscale
    image = cv2.imread(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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

    img=cv2.imread(image)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    phase_spectrum = np.angle(dft_shift)
    return phase_spectrum



if __name__ == "__main__":

    imageCT = 'Dataset/BiometricMedicalGrayscaleImages/CT.jpg'
    imageFaceThermogram = 'Dataset/BiometricMedicalGrayscaleImages/face_thermogram.png'
    imageFinger = 'Dataset/BiometricMedicalGrayscaleImages/finger.png'
    imageIris = 'Dataset/BiometricMedicalGrayscaleImages/iris.png'
    imageMR = 'Dataset/BiometricMedicalGrayscaleImages/MR.jpg'
    imagePET = 'Dataset/BiometricMedicalGrayscaleImages/PET.png'
    imageThyroid = 'Dataset/BiometricMedicalGrayscaleImages/Thyroid.tif'
    imageXRay = 'Dataset/BiometricMedicalGrayscaleImages/XRay.png'

    imgRead = cv2.imread(imageFaceThermogram)
    imgSpectrumModule = obtainSpectrumModule(imageFaceThermogram)
    imgSpectrumPhase = obtainSpectrumPhase(imageFaceThermogram)
    calculateEntropy(imageFaceThermogram)

    # Calculate histogram for one of the images
    hist_data = cv2.calcHist([imgRead], [0], None, [256], [0, 256])

    # Normalize the histogram
    hist_data /= hist_data.sum()

    plt.subplot(2, 2, 1)
    plt.imshow(imgRead, cmap='gray')
    plt.title('Original Image')
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



