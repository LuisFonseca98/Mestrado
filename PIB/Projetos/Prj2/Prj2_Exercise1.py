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
print('ImageCT', imageCT)
displayPlotImages('Original Image',imageCT, pathToSavePictures + 'Plot ImageCT.png')
print("")
print('imageFaceThermogram', imageFaceThermogram)
displayPlotImages('Original Image',imageFaceThermogram, pathToSavePictures + 'Plot Face Thermogram.png')
print("")
print('imageFinger', imageFinger)
displayPlotImages('Original Image',imageFinger, pathToSavePictures + 'Plot Finger.png')
print("")
print('imageIris', imageIris)
displayPlotImages('Original Image',imageIris, pathToSavePictures + 'Plot Iris.png')
print("")
print('imageMR', imageMR)
displayPlotImages('Original Image',imageMR, pathToSavePictures + 'Plot MR.png')
print("")
print('imagePET', imagePET)
displayPlotImages('Original Image',imagePET, pathToSavePictures + 'Plot PET.png')
print("")
print('imageThyroid', imageThyroid)
displayPlotImages('Original Image',imageThyroid, pathToSavePictures + 'Plot Thyroid.png')
print("")
print('imageFinger', imageFinger)
displayPlotImages('Original Image',imageXRay, pathToSavePictures + 'Plot Finger.png')

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
print("")
print('Brightness Image imageCT_Brightness', imageCT_Brightness)
displayPlotImages('Brightness Image',imageCT_Brightness, pathToSavePictures + 'Plot_CT_Brightness_Image.png')

print("")
print('Brightness Image imageFaceThermogram_Brightness', imageFaceThermogram_Brightness)
displayPlotImages('Brightness Image',imageFaceThermogram_Brightness, pathToSavePictures + 'Plot_Thermogram_Brightness_Image.png')

print("")
print('Brightness Image imageFaceimageFinger_Brightness', imageFinger_Brightness)
displayPlotImages('Brightness Image',imageFinger_Brightness, pathToSavePictures + 'Plot_Finger_Brightness_Image.PNG')

print("")
print('Brightness Image imageFaceimageimageIris_Brightness', imageIris_Brightness)
displayPlotImages('Brightness Image',imageIris_Brightness, pathToSavePictures + 'Plot_Iris_Brightness_Image.png')

print("")
print('Brightness Image imageFaceimageimageimageMR_Brightness', imageMR_Brightness)
displayPlotImages('Brightness Image',imageMR_Brightness, pathToSavePictures + 'Plot_MR_Brightness_Image.png')

print("")
print('Brightness Image imageFaceimageimageimageimagePET_Brightness', imagePET_Brightness)
displayPlotImages('Brightness Image',imagePET_Brightness, pathToSavePictures + 'Plot_PET_Brightness_Image.png')

print("")
print('Brightness Image imageFaceimageThyroid_Brightness', imageThyroid_Brightness)
displayPlotImages('Brightness Image',imageThyroid_Brightness, pathToSavePictures + 'Plot_Thyroid_Brightness_Image.png')

print("")
print('Brightness Image imageFaceimageXRay_Brightness', imageXRay_Brightness)
displayPlotImages('Brightness Image',imageXRay_Brightness, pathToSavePictures + 'Plot_XRay_Brightness_Image.png')
