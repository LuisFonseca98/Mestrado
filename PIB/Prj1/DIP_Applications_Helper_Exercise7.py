from fileinput import filename
import cv2
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np


# """
# Shows the orignal image
# TO DEBUG PURPOSE ONLY
# """
# def showOriginalImage(image):
	
# 	imgRead = cv2.imread(image)
	
# 	print('Orinal Image Matrix', imgRead)
# 	cv2.imshow('Original Image', imgRead)

# 	plt.title('Original Image Histogram')
# 	plt.hist(imgRead.ravel(),256,[0,256])  
# 	plt.show() 
	
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

# Filesystem paths
baseSavePath = os.path.join(os.getcwd(), 'SavedPictures')
blurredImagesPath = os.path.join(baseSavePath, 'Blur')
negativeImagesPath = os.path.join(baseSavePath, 'Negative')
contrastImagesPath = os.path.join(baseSavePath, 'ContrastEnhancement')
featureDetectionImagesPath = os.path.join(baseSavePath, 'FeatureDetection')
jpegQualityLevelsPath = os.path.join(baseSavePath, 'JpegQualityLevels')

# File extensions
jpegExtension = '.jpg'
pngExtension = '.png'

# Filename and directory suffixes
blurredFilenameSuffix = '_blurred'
negativeFilenameSuffix = '_negative'
histogramEqualizedFilenameSuffix = '_histogramEqualized'
featureDetectedFilenameSuffix = '_faceFeatures'
jpegQualityLevelFilenameSuffix = '_jpegQualityLevel_'
jpegQualityLevelDirectorySuffix = '_jpegQualityLevels'

"""
Making sure required destination directories for results exist
"""
def setupFilesystem():
	# base path for destination directories
	if not os.path.exists(baseSavePath):
		os.mkdir(baseSavePath)
		
	# destination directories for DIP operation results
	for path in [blurredImagesPath, negativeImagesPath, contrastImagesPath, featureDetectionImagesPath, jpegQualityLevelsPath]:
		if not os.path.exists(path):
			os.mkdir(path)


"""
Saving cv image in JPEG and PNG formats
"""
def saveImageInJpgAndPngFormats(image, fileName, path):
	cv2.imwrite(os.path.join(path, fileName + jpegExtension), image)
	cv2.imwrite(os.path.join(path, fileName + pngExtension), image)



# def stackImages(images):
# 	#	2 images
# 	#	convert all to color
# 	images = [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.shape[2] == 1 else image for image in images]
# 	channelCount = 3
	
# 	# stacked dimentions
# 	height = max([image.shape[0] for image in images])
# 	#width = sum([image.shape[1] for image in images])
	
# 	conformed = []
# 	for image in images:
# 		vOffset = int((height - image.shape[0]) / 2)
# 		extraLine = (height - image.shape[0]) % 2
# 		conformed.append(np.pad(image, [(vOffset, vOffset + extraLine), (0, 0), (0, 0)], mode = 'constant'))
		
# 	#dest = np.concatenate([conformed], axis=0)
# 	return np.hstack(conformed)



"""
Displaying results using Maptplotlib
Contents of 'displayData' are taken as image data except for entries which indices are contained in 'plotIndices'
in those cases, the entries are taken as plot data
"""
def displayResult(displayData, titles, nRows = 1, plotIndices = []):
	# display images with matplotlib
	nCols = int(len(displayData) / nRows)
	for i, image in enumerate(displayData):
		plt.subplot(nRows, nCols, i+1)
		plt.title(titles[i])
		if i in plotIndices:
			plt.plot(image)
		else:
			plt.imshow(cv2.cvtColor(displayData[i], cv2.COLOR_BGR2RGB))
			plt.xticks([])
			plt.yticks([])
		
	plt.show()

	
"""
Blurring an image
"""
def blurImage(imagePath):
	blurKernelSize = (25, 25)

	# extract from path
	img = cv2.imread(imagePath)
	_, fileNameExt = os.path.split(imagePath)
	fileName, _ = os.path.splitext(fileNameExt)

	#perform operation
	blurredImage = cv2.blur(img, (25, 25), 0)

	# display and save results
	displayResult([img, blurredImage], ['original', 'blurred'])
	saveImageInJpgAndPngFormats(blurredImage, fileName + blurredFilenameSuffix, blurredImagesPath)
	
	return blurredImagesPath


"""
Identity hiding by blurring faces in image
"""
def blurImageFaces(imagePath):
	blurKernelSize = (25, 25)

	# extract from path
	img = cv2.imread(imagePath)
	_, fileNameExt = os.path.split(imagePath)
	fileName, _ = os.path.splitext(fileNameExt)
	resultImage = img.copy()

	# Setting up face detection classifier. Load pretrained cascade Haar classifier from file 
	faceClassifier = cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')
	
	# confidence thresholds for classifier results: detections below this values get rejected
	# this value was obtained experimentally
	faceConfidenceThreshold = 0.0

	# convert to grayscale for apllying the classifiers
	grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# get a set of rectangles arround faces in the image
	faces, _, confidenceLevels = faceClassifier.detectMultiScale3(grayImage, outputRejectLevels = True)

	# for every face detected
	for i, (x, y, w, h) in enumerate(faces):
		if confidenceLevels[i] > faceConfidenceThreshold:
			# blur face detected area
			blurredFace = cv2.GaussianBlur(img[y:y+h, x:x+w, :], blurKernelSize, 0)
			resultImage[y:y+h, x:x+h, :] = blurredFace

	# display and save results
	displayResult([img, resultImage], ['original', 'blurred'])
	saveImageInJpgAndPngFormats(resultImage, fileName + blurredFilenameSuffix, blurredImagesPath)
	
	return blurredImagesPath
	
"""
Composes the negative of an image
"""
def negativeImage(imagePath):
	# extract from path
	img = cv2.imread(imagePath)
	_, fileNameExt = os.path.split(imagePath)
	fileName, _ = os.path.splitext(fileNameExt)

	# perform operation
	# negative image is produced by subtracting each color component of each pixel to the maximum value for the channel
	colorChannel_maxValue = 255
	negative = colorChannel_maxValue-img
	
	# display and save results
	imgs = [img, negative]			#[imgConverted, grayImagePath, colorNegative, grayImageNegative]
	title = ['original', 'negative']	#['coloured', 'gray', 'coloured-negative', 'gray-negative']
	displayResult(imgs, title)	
	saveImageInJpgAndPngFormats(negative, fileName, negativeImagesPath)

	return negativeImagesPath


"""
Changes the contrast of an image
@alpha: Contrast control 
		To lower the contrast, use 0 < alpha < 1 
		And for higher contrast use alpha > 1.
		
@beta: brightness control
		A good range for brightness value is [-127, 127]
"""

def improveContrast(imagePath):
	# extract from path
	img = cv2.imread(imagePath)
	_, fileNameExt = os.path.split(imagePath)	
	fileName, _ = os.path.splitext(fileNameExt)

	# performing operation
	# equalizing BGR channels individually
	eqB = cv2.equalizeHist(img[:, :, 0])
	eqG = cv2.equalizeHist(img[:, :, 1])
	eqR = cv2.equalizeHist(img[:, :, 2])
	equalizedImage = cv2.merge([eqB, eqG, eqR])
	
	# calculate histograms of grayscale levels for display
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	histOriginal = cv2.calcHist([gray], [0], None, [256], [0,256])
	gray = cv2.cvtColor(equalizedImage, cv2.COLOR_BGR2GRAY)
	histNew = cv2.calcHist([gray], [0], None, [256], [0,256])

	# displaying and saving results
	displayResult([img, equalizedImage, histOriginal, histNew], ['original image', 'improved image', 'original brightnes hist.', 'enhanced brightness hist'], 2, [2, 3])
	saveImageInJpgAndPngFormats(equalizedImage, fileName, contrastImagesPath)
	
	return contrastImagesPath
	


"""
Detects face features in images
"""
def faceDetectionImages(imagePath):

	# extracting from path 
	img = cv2.imread(imagePath)
	_, fileNameExt = os.path.split(imagePath)
	fileName, _ = os.path.splitext(fileNameExt)
	
	# Setting up classifiers. Load pretrained cascade Haar classifiers from file 
	#	face classifier for narrowing down the search for mouths and eyes
	faceClassifier = cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')
	#	classifier for detecting human eyes
	eyeClassifier = cv2.CascadeClassifier('Classifiers/haarcascade_eye_tree_eyeglasses.xml')
	#	classifier for detecting human mouths
	mouthClassifier = cv2.CascadeClassifier('Classifiers/haarcascade_smile.xml')
	
	# confidence thresholds for classifier results: detections below these values get rejected
	# these values were obtained experimentally
	faceConfidenceThreshold = 0.0
	eyeConfidenceThreshold = 0.0
	mouthConfidenceThreshold = 2.0

	# convert to grayscale for apllying the classifiers
	grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# get a set of rectangles arround faces in the image
	faces, _, confidenceLevels = faceClassifier.detectMultiScale3(grayImage, outputRejectLevels = True)

	# for every face detected
	for i, (x, y, w, h) in enumerate(faces):
		if confidenceLevels[i] > faceConfidenceThreshold:
			# extracting face area
			roi = grayImage[y:y+h, x:x+w]
			
			#searching for eyes within face area
			eyes, _, levels = eyeClassifier.detectMultiScale3(roi, minNeighbors=1, outputRejectLevels = True)
			for j, (ex,ey,ew,eh) in enumerate(eyes):
				if(levels[j] > eyeConfidenceThreshold):
					cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
					
			#searching for mouths within face area
			mouths, _, levels = mouthClassifier.detectMultiScale3(roi, minNeighbors=3, outputRejectLevels = True)
			for j, (mx,my,mw,mh) in enumerate(mouths):
				if(levels[j] > mouthConfidenceThreshold):
					cv2.rectangle(img, (x + mx, y + my), (x + mx + mw, y + my + mh), (0, 0, 255), 2)

	# displaying and saving results
	displayResult([img], ['eyes and mouth detection'])
	saveImageInJpgAndPngFormats(img, fileNameExt, featureDetectionImagesPath)
				
	return featureDetectionImagesPath



"""
Saves the image with different quality
JPG_Quality: 0 - 100 (higher the better quality). Default is 95
PNG_Quality: 0 - 9 (higher means a smaller size and longer compression time)
"""    
def savePictureJPEGCompressionLevels(imagePath, levels = range(25, 101, 25)):
	#extracting from path
	img = cv2.imread(imagePath)		
	_, fileNameExt = os.path.split(imagePath)
	fileName, _ = os.path.splitext(fileNameExt)

	# composing destination directory path
	destPath = os.path.join(jpegQualityLevelsPath, fileName  + jpegQualityLevelDirectorySuffix)
	
	# if destination exists, delete it along with content. Then create it empty.
	if os.path.exists(destPath) and os.path.isdir(destPath):
		shutil.rmtree(destPath)
	
	os.mkdir(destPath)
	
	# saving quality level images to the path determined and showing results
	filePaths = []
	images = []
	titles = []
	for i, level in enumerate(levels):
		# computing file path
		filePaths.append(os.path.join(destPath, f'{fileNameExt}_jpegQualityLevel_{level}{jpegExtension}'))
		# saving quality level image
		cv2.imwrite(filePaths[i], img, [(int)(cv2.IMWRITE_JPEG_QUALITY), level])
		# gathering image and title for display
		images.append(cv2.imread(filePaths[i]))
		titles.append(f'JPEG Quality level: {level}')
		
	# showing first 4 levels
	displayResult(images[:4], titles[:4], 2)

	return destPath



if __name__ == "__main__":

	#####DATASET#########
	image = 'Dataset/face1.jpg'
	image2 = 'Dataset/face2.bmp'
	image3 = 'Dataset/face3.jpg'
	image4 = 'Dataset/face4.jpg'
	image5 = 'Dataset/face5.jpg'
	image6 = 'Dataset/face6.jpg'
	image7 = 'Dataset/face7.jpg'
	imageLena = 'Dataset/Lena.jpg'
	imageGreyLena = 'Dataset/lenaGray.jpg'
	pathToSavePictures = 'SavedPictures/'
	
	
	#####VARIABELS TO CHANGE VALUES########
	jpg_quality = 25
	png_quality = 1
	
	####METHOD CALL#####
	
	#showOriginalImage(imageLena)
	#blurImage(imageLena,pathToSavePictures+'LenaBlurred.jpg')
	#negativeImage(imageLena,pathToSavePictures,False)
	#changeContrast(image2,pathToSavePictures + 'ImageAfterContrast.bmp')
	#faceDetectionImages(imageLena)
	#savePictureJPEGCompression(pathToSavePictures+'LenaJPG_' + str(jpg_quality) + '.jpg',imageLena,jpg_quality=jpg_quality)
	#savePictureJPEGCompression(pathToSavePictures+'LenaPNG_' + str(png_quality) + '.png',imageLena,png_quality=png_quality)