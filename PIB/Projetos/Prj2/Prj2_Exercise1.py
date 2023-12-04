from Prj2_Exercise1_Helper import *

# init images dataset
imageCT = 'Dataset/BiometricMedicalGrayscaleImages/CT.jpg'
imageFaceThermogram = 'Dataset/BiometricMedicalGrayscaleImages/face_thermogram.png'
imageFinger = 'Dataset/BiometricMedicalGrayscaleImages/finger.png'
imageIris = 'Dataset/BiometricMedicalGrayscaleImages/iris.png'
imageMR = 'Dataset/BiometricMedicalGrayscaleImages/MR.jpg'
imagePET = 'Dataset/BiometricMedicalGrayscaleImages/PET.png'
imageThyroid = 'Dataset/BiometricMedicalGrayscaleImages/Thyroid.tif'
imageXRay = 'Dataset/BiometricMedicalGrayscaleImages/XRay.png'

pathToSavePictures = 'SavedImages/'
brightnessImage = 'Brightness'
contrastImage = 'Contrast'

#display original images without DIP
displayPlotImages('Original Image',imageCT)
displayPlotImages('Original Image',imageFaceThermogram)
displayPlotImages('Original Image',imageFinger)
displayPlotImages('Original Image',imageIris)
displayPlotImages('Original Image',imageMR)
displayPlotImages('Original Image',imagePET)
displayPlotImages('Original Image',imageThyroid)
displayPlotImages('Original Image',imageXRay)

#apply brightness to an image
applyBrightness(pathToSavePictures + brightnessImage + '_CT_Image.jpg',imageCT)
applyBrightness(pathToSavePictures + brightnessImage + '_Thermogram_Image.png',imageFaceThermogram)
applyBrightness(pathToSavePictures + brightnessImage + '_Finger_Image.png',imageFinger)
applyBrightness(pathToSavePictures + brightnessImage + '_Iris_Image.png',imageIris)
applyBrightness(pathToSavePictures + brightnessImage + '_MR_Image.jpg',imageMR)
applyBrightness(pathToSavePictures + brightnessImage + '_PET_Image.jpg',imagePET)
applyBrightness(pathToSavePictures + brightnessImage + '_Thyroid_Image.tiff',imageThyroid)
applyBrightness(pathToSavePictures + brightnessImage + '_XRay_Image.png',imageXRay)

#get the new images with brightness applied
imageCT_Brightness = pathToSavePictures + 'Brightness_CT_Image.jpg'
imageFaceThermogram_Brightness = pathToSavePictures + 'Brightness_Thermogram_Image.png'
imageFinger_Brightness = pathToSavePictures + 'Brightness_Finger_Image.png'
imageIris_Brightness = pathToSavePictures + 'Brightness_Iris_Image.png'
imageMR_Brightness = pathToSavePictures + 'Brightness_MR_Image.jpg'
imagePET_Brightness = pathToSavePictures + 'Brightness_PET_Image.jpg'
imageThyroid_Brightness = pathToSavePictures + 'Brightness_Thyroid_Image.tiff'
imageXRay_Brightness = pathToSavePictures + 'Brightness_XRay_Image.png'

#show plot image with brightness
displayPlotImages('Brightness Image',imageCT_Brightness)
displayPlotImages('Brightness Image',imageFinger_Brightness)
displayPlotImages('Brightness Image',imageIris_Brightness)
displayPlotImages('Brightness Image',imageMR_Brightness)
displayPlotImages('Brightness Image',imagePET_Brightness)
displayPlotImages('Brightness Image',imageThyroid_Brightness)
displayPlotImages('Brightness Image',imageXRay_Brightness)
