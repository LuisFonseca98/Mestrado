from fileinput import filename
from pkgutil import ModuleInfo
import cv2
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import math


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
grayNoisyImagesSourcePath = "images/gray"
rgbNoisyImagesSourcePath = "images/rgb"

# File extensions
jpegExtension = '.jpg'
pngExtension = '.png'
bmpExtension = '.bmp'
txtFileExtesion = '.txt'

# Filename and directory suffixes
moduloImageSuffix = '_modulo'
maskImageSuffix = '_mask'



"""
Creates path if 'path' doesn't exist
"""
def createDirIfNotExists(path):
	if not os.path.exists(path):
		os.mkdir(path)



"""
Displaying results using Maptplotlib
Contents of 'displayData' are taken as image data except for entries which indices are contained in 'plotIndices'
in those cases, the entries are taken as plot data
"""
def displayResult(displayData, titles, nRows = 1, plotIndices = []):
	# display images with matplotlib
	nCols = int(math.ceil(len(displayData) / nRows))
	for i, image in enumerate(displayData):
		plt.subplot(nRows, nCols, i+1)
		plt.title(titles[i])
		if i in plotIndices:
			plt.plot(image)
		else:
			if len(displayData[i].shape) < 3:
				# less than 3 channel images
				plt.imshow(displayData[i], cmap='gray')
			else:
				# 3 channel images
				plt.imshow(cv2.cvtColor(displayData[i], cv2.COLOR_BGR2RGB))
			
			plt.xticks([])
			plt.yticks([])
		
	plt.show(block=True)



#	Returns euclidean distance between points ('x', 'y') and ('shape[1]/2', 'shape[0]/2'), the latter
# being the center of the shape described by 'shape'.
def squaredDistFromCenter(x, y, shape):
	return (x - shape[1]/2)**2 + (y - shape[0]/2)**2

#	Returns mean squared error between 'mat1' and 'mat2'
def meanSquaredError(mat1, mat2):
	return np.square(mat1-mat2).mean()

#	Returns mean absolute error between 'mat1' and 'mat2'
def meanAbsoluteError(mat1, mat2):
	return np.absolute(mat1-mat2).mean()

#	Returns signal to noise ratio given a signal and a noisy signal
def signalToNoiseRatio(signal, noisySignal):
	return 	10.0*np.log10(np.square(signal).sum()/np.square(signal-noisySignal).sum())

def renormalizeUInt8(image):
	return (255.0*(image-image.min())/(image.max()-image.min()))

def bgr2hsi(img):
	# convert components from [0..255]	to [0.0..1.0]
	B, G, R = img[:, :, 0]/255.0, img[:, :, 1]/255.0, img[:, :, 2]/255.0
	
	# calculations in advance
	sBGR = B + G + R
	R_B = R - B
	R_G = R - G
	G_B = G - B

	# - Hue - 
	# calculate theta
	#Th = np.arccos(np.clip((R - 0.5*(B + G))/np.sqrt(B**2 + G**2 + R**2 - B*G - B*R - G*R), -1.0, 1.0))
	#Th = np.arccos(np.clip(0.5*(R_B + R_G)/np.sqrt(R_G**2 + R_B*G_B), -1.0, 1.0))
	num = 0.5*(R_B + R_G)
	denom = R_G**2 + R_B*G_B
	Th = np.array([[np.arccos(np.clip(num[y, x]/np.sqrt(denom[y, x]), -1, 1)) if denom[y, x]!=0 else np.arccos(1.0) for x in range(B.shape[1])] for y in range(B.shape[0])])
			
	# function for Hue
	hfunc = lambda x, y : Th[y, x] if G[y, x] >= B[y, x] else 2.0*np.pi - Th[y, x]
	# function for Saturation
	sfunc = lambda x, y: 1.0 - 3.0*np.min([B[y, x], G[y, x], R[y, x]], 0)/sBGR[y, x] if sBGR[y, x] > 0.0 else 0.0
	# function for Intensity
	ifunc = lambda x, y: sBGR[y, x]/3.0
	
	# calculate HSI
	HSI = np.array([[[hfunc(x, y), sfunc(x, y), ifunc(x, y)] for x in range(Th.shape[1])] for y in range(Th.shape[0])])
	# convert to [0..255]
	HSI[:, :, 0] = 255.0*HSI[:, :, 0]/(2.0*np.pi)
	HSI[:, :, 1:] = 255.0*HSI[:, :, 1:]

	return HSI.astype(np.uint8)

def hsi2bgr(img):
	# convert H to radians
	H = 2.0*np.pi*img[:, :, 0]/255.0	
	
	# convert S and I from [0..255] to [0.0..1.0]
	S, I = img[:, :, 1]/255.0, img[:, :, 2]/255.0
	
	# initialize components
	B = np.zeros(img.shape[:2])
	G = np.zeros(img.shape[:2])
	R = np.zeros(img.shape[:2])
	
	# calculate in advance
	pi1_3 = np.pi/3.0
	pi2_3 = 2.0*np.pi/3.0
	pi4_3 = 4.0*np.pi/3.0
	pi5_3 = 5.0*np.pi/3.0
	
	for y, row in enumerate(H):
		for x, h in enumerate(row):
			i = I[y, x]
			s = S[y, x]
			
			t0 = lambda: i*(1.0 - s)
			t1 = lambda p1, p2 : i*(1.0 + s*np.cos(h-p1)/np.cos(p2 - h))
			t2 = lambda p1, p2 : i*(1.0 + s*(1.0 - np.cos(h-p1)/np.cos(p2 - h)))
		
			if(h < pi2_3):
				p1 = 0.0
				p2 = pi1_3
				B[y, x] = t0()
				G[y, x] = t2(p1, p2)
				R[y, x] = t1(p1, p2)
			elif h < pi4_3:
				p1 = pi2_3
				p2 = np.pi
				B[y, x] = t2(p1, p2)
				G[y, x] = t1(p1, p2)
				R[y, x] = t0()
			else: # 'h >= pi4_3'
				p1 = pi4_3
				p2 = pi5_3
				B[y, x] = t1(p1, p2)
				G[y, x] = t0()
				R[y, x] = t2(p1, p2)		
	
	# convert back to [0..255]
	return np.round(np.clip(cv2.merge([B, G, R])*255.0, 0, 255)).astype(np.uint8)
	
	
			
def unsharpMasking(image, k, d, n):
	# get spectrum	
	modulo, phase = getModuloAndPhase(image)
	# make low pass butterworth mask
	#d = 35
	#n = 4
	filterMask = makeButterworthLowPassMask(d, n, image.shape)
	# cast image to float32
	image = image.astype(np.float32)
	# calculate blurred version from low passing
	blurredVersion = recombineModuloAndPhase(modulo*filterMask, phase)
	# calculate unsharp mask
	unsharpMask = image - blurredVersion
	# add proportion 'k' of mask
	result = image + k*unsharpMask
	# renormalize and convert to 'uint8'
	result = (255.0*(result-result.min())/(result.max()-result.min())).astype(np.uint8)
	
	return result, unsharpMask, blurredVersion


def compareRgbAndHsiHistograms(image1, image2):
	bgr1 = cv2.split(image1)
	bgr2 = cv2.split(image2)
	hsi1 = cv2.split(bgr2hsi(image1))
	hsi2 = cv2.split(bgr2hsi(image2))
	displayResult([cv2.calcHist(channel, [0], None, [256], [0, 256]) for channel in bgr1+bgr2+hsi1+hsi2], 'BGRBGRHSIHSI', 4, list(range(12)))
	
	
'''
	DFT {
'''
# Given an image calculates discrete fourier transform spectrum modulo and phase and returns them
def getModuloAndPhase(image):
	#padded = np.zeros((image.shape[0]*2, image.shape[1]*2))
	#padded[:image.shape[0], :image.shape[1]] = image
	padded = image
	dft = cv2.dft(np.float32(padded), flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	return cv2.cartToPolar(dft_shift[:,:,0], dft_shift[:,:,1])	


#	Performs inverse discrete fourier transform on spectrum modulo and phase given and returns 
# the resulting image
def recombineModuloAndPhase(modulo, phase):
	real, imag = cv2.polarToCart(modulo, phase)
	back = cv2.merge([real, imag])
	back_ishift = np.fft.ifftshift(back)
	img_back = cv2.idft(back_ishift) #[:int(modulo.shape[0]/2),:int(modulo.shape[1]/2)]
	img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
	min, max = np.amin(img_back, (0,1)), np.amax(img_back, (0,1))
	return cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	
#	Given a d.f.t. spectrum module, performs the 'log' operation on its values and renormalizes the
# result for viewing. Returns a the resulting image.
def getModuloDisplayImage(modulo):
	disp = np.log(modulo)
	disp = np.clip(255*disp/disp.max(), 0.0, 255.0)
	return disp.astype(np.uint8)

'''
	} DFT
'''

'''
 Mask {
 '''
 #	Returns a new butterworth band reject mask (d0 is the cutoff frequency)
def makeButterworthBandRejectMask(d0, bandWidth, n, shape):
	mask = np.zeros((shape[0], shape[1]), np.float32)

	for j in range(0, shape[0]):  # image width
		for i in range(0, shape[1]):  # image height
			#radius = max(1, math.sqrt(math.pow((i - centre[0]), 2.0) + math.pow((j - centre[1]), 2.0)))
			d = max(1, math.sqrt(squaredDistFromCenter(i, j, shape)))
			try:
				mask[j, i] = 1 / (1 + math.pow(d*bandWidth/(d**2 - d0**2), 2*n))
			except:
				mask[j, i] = 1
			
	return mask


#	Returns a new butterworth high pass mask (d0 is the cutoff frequency)
def makeButterworthHighPassMask(d0, n, shape):
	mask = np.zeros((shape[0], shape[1]), np.float32)

	for i in range(shape[1]):  # image width
		for j in range(shape[0]):  # image height
			d = max(1, math.sqrt(squaredDistFromCenter(i, j, shape)))
			mask[j, i] = 1 / (1 + math.pow(d0/d, 2*n))
			
	return mask
	

#	Returns a new butterworth low pass mask (d0 is the cutoff frequency)
def makeButterworthLowPassMask(d, n, shape):
	return 1.0 - makeButterworthHighPassMask(d, n, shape)


#	Generates a mask with values based on their position indices as given by 'func'(int, int) : bool
def makeMask(func, shape):
	return [[1.0 if func(x, y) else 0.0 for x in range(shape[0])] for y in range(shape[1])]

#	Returns a new ideal low pass mask for frequencies in between low and high
def makeIdealBandRejectMask(low, high, shape):
	func = lambda x, y : squaredDistFromCenter(x, y, shape) < low**2 or squaredDistFromCenter(x, y, shape) > high**2
	return np.array(makeMask(func, shape))

#	Returns a new ideal low pass mask for frequencies in between low and high
def makeIdealLowPassMask(t, shape):
	func = lambda x, y : squaredDistFromCenter(x, y, shape) < t**2
	return np.array(makeMask(func, shape))

#	Given a filter mask, returns a version suitable for viewing
def getMaskDisplayImage(mask):
	disp = np.clip(255*mask/mask.max(), 0.0, 255.0)
	return disp.astype(np.uint8)

def viewMaskProfile(mask):
	displayResult([mask[int(mask.shape[0]/2), int(mask.shape[1]/2):]], ["filter profile"], 1, [0])

'''
	} Mask
'''

#	Generates lines of a case report base on 'original', 'noisy', 'result' and 'caseName'.
#	Report will contain mean squared error, mean absolute error and signal to noise ratio.
def generateReportLines(caseName, original, noisy, modulo, mask, masked, result):
		return [f"Case: {caseName}\n",
		"\n- Original vs Cleansed\n",
		f"mean squared error : {meanSquaredError(original, result)}\n",
		f"mean absolute error : {meanAbsoluteError(original, result)}\n",
		
		"\n- Cleansed vs Noisy\n",
		f"mean squared error : {meanSquaredError(result, noisy)}\n",
		f"mean absolute error : {meanSquaredError(result, noisy)}\n",

		f"\n- signal to ratio : {signalToNoiseRatio(result, noisy)}\n",

		"\n- Original vs Noisy\n",
		f"mean squared error : {meanSquaredError(original, noisy)}\n",
		f"mean absolute error : {meanAbsoluteError(original, noisy)}\n"]

#	Writes images involved in an image treatment case, plus a human readable report.
# Returns the report lines as a list of strings
def writeCaseData(caseName, outputPath, original, noisy, modulo, mask, masked, result):
	# write report	
	reportLines = generateReportLines(caseName, original, noisy, modulo, mask, masked, result)	
	with(open(os.path.join(outputPath, caseName + '_report' + txtFileExtesion), 'w+')) as file:		
		file.writelines(reportLines)
	# write images
	writeCaseImages(caseName, outputPath, modulo, mask, masked, result)
	# return report lines
	return reportLines
		
# Writes images involved in a image treatment case.
def writeCaseImages(caseName, outputPath, modulo, mask, masked, result):
	# modulo
	cv2.imwrite(os.path.join(outputPath, caseName + moduloImageSuffix + bmpExtension), getModuloDisplayImage(modulo))
	# mask
	cv2.imwrite(os.path.join(outputPath, caseName + maskImageSuffix + bmpExtension), getMaskDisplayImage(mask))
	# modulo after applying the mask	
	cv2.imwrite(os.path.join(outputPath, caseName + '_masked' + moduloImageSuffix + bmpExtension), getModuloDisplayImage(masked))
	# image reconstructed with new modulo
	cv2.imwrite(os.path.join(outputPath, caseName + '_result' + moduloImageSuffix + bmpExtension), result)

#	Displays images involved in a case in a window, using matplotlib
def displayCaseImages(caseName, original, noisy, modulo, mask, masked, result):
	displayResult([original, result, noisy, getModuloDisplayImage(modulo), getModuloDisplayImage(masked), getMaskDisplayImage(mask)], 
			   ["original", "cleansed", caseName, "modulo", "masked modulo", "mask"], 2)

#	Case: "finger_1.bmp"
#	Returns tuple: (noisy image, noisy image modulo, mask, cleansed image modulo, cleansed image)
def caseFinger1():
	# loading the finger_1 image and converting it to grayscale
	finger = cv2.imread(os.path.join(grayNoisyImagesSourcePath, "finger_1.bmp"));
	finger = cv2.cvtColor(finger, cv2.COLOR_BGR2GRAY)
	# getting dft modulo and phase from image
	mod, phase = getModuloAndPhase(finger)
	# making a band reject filter mask
	mask = makeIdealBandRejectMask(5, 21, finger.shape) #*makeIdealLowPassMask(42, finger.shape)
	# applying the mask to the modulo
	masked = (mod*mask).astype(np.float32)
	# idft reconstructing the image 
	result = recombineModuloAndPhase(masked, phase)
	# returning images involved
	return (finger, mod, mask, masked, result)

#	Case: "finger_2.bmp"
#	Returns tuple: (noisy image, noisy image modulo, mask, cleansed image modulo, cleansed image)
def caseFinger2():
	# loading the finger_1 image and converting it to grayscale
	finger = cv2.imread(os.path.join(grayNoisyImagesSourcePath, "finger_2.bmp"));
	finger = cv2.cvtColor(finger, cv2.COLOR_BGR2GRAY)
	# getting dft modulo and phase from image
	mod, phase = getModuloAndPhase(finger)
	# generating a band reject filter mask with 
	mask = makeIdealLowPassMask(30, finger.shape)
	# applying the mask to the modulo
	masked = (mod*mask).astype(np.float32)
	# idft reconstructing the image
	result = recombineModuloAndPhase(masked, phase)
	# returning it along with the initial image
	return (finger, mod, mask, masked, result)

#	Case: "finger_3.bmp"
#	Returns tuple: (noisy image, noisy image modulo, mask, cleansed image modulo, cleansed image)
def caseFinger3():
	# loading the finger_1 image and converting it to grayscale
	finger = cv2.imread(os.path.join(grayNoisyImagesSourcePath, "finger_3.bmp"));
	finger = cv2.cvtColor(finger, cv2.COLOR_BGR2GRAY)
	# getting dft modulo and phase from image
	mod, phase = getModuloAndPhase(finger)
	#loading the notch filter mask from file
	mask = cv2.cvtColor(cv2.imread("finger3mask.bmp"), cv2.COLOR_BGR2GRAY)/255.0;
	# applying the mask to the modulo
	masked = (mod*mask).astype(np.float32)
	# idft reconstructing the image
	result = recombineModuloAndPhase(masked, phase)
	# returning images involved
	return (finger, mod, mask, masked, result)

#	Case: "finger_4.bmp"
#	Returns tuple: (noisy image, noisy image modulo, mask, cleansed image modulo, cleansed image)
def caseFinger4():
	# loading the finger_1 image and converting it to grayscale
	finger = cv2.imread(os.path.join(grayNoisyImagesSourcePath, "finger_4.bmp"));
	finger = cv2.cvtColor(finger, cv2.COLOR_BGR2GRAY)
	# getting dft modulo and phase from image
	mod, phase = getModuloAndPhase(finger)
	#loading the notch filter mask from file
	mask = cv2.cvtColor(cv2.imread("finger4mask.bmp"), cv2.COLOR_BGR2GRAY)/255.0;
	# applying the mask to the modulo
	masked = (mod*mask).astype(np.float32)
	# idft reconstructing the image
	result = recombineModuloAndPhase(masked, phase)
	# returning images involved
	return (finger, mod, mask, masked, result)

#	Case: "finger_5.bmp"
#	Returns tuple: (noisy image, noisy image modulo, mask, cleansed image modulo, cleansed image)
def caseFinger5():
	# loading the finger_1 image and converting it to grayscale
	finger = cv2.imread(os.path.join(grayNoisyImagesSourcePath, "finger_5.bmp"));
	finger = cv2.cvtColor(finger, cv2.COLOR_BGR2GRAY)
	# getting dft modulo and phase from image
	mod, phase = getModuloAndPhase(finger)
	#loading the notch filter mask from file
	mask = cv2.cvtColor(cv2.imread("finger5mask.bmp"), cv2.COLOR_BGR2GRAY)/255.0;
	# applying the mask to the modulo
	masked = (mod*mask).astype(np.float32)
	# idft reconstructing the image
	result = recombineModuloAndPhase(masked, phase)
	# returning images involved
	return (finger, mod, mask, masked, result)

def generateImages():

	# Face Images - all cases treated with median spatial filter

	# load main face image
	face = cv2.imread(os.path.join(grayNoisyImagesSourcePath, "face.bmp"))
	face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)	
	# for every noisy face image...
	caseNames = ["face_1", "face_2", "face_3", "face_4", "face_5"]
	lowPassD = [50, 40, 30, 20, 10]
	unsharpD = [20,  16,  12,  8,  4]
	for i, caseName in enumerate(caseNames):
		# load noisy image
		noisy = cv2.imread(os.path.join(grayNoisyImagesSourcePath, caseName + bmpExtension))
		noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
		# create output folder for case
		createDirIfNotExists(caseName)
		# write original and noisy image into output folder
		cv2.imwrite(os.path.join(caseName, "face" + bmpExtension), face)
		cv2.imwrite(os.path.join(caseName, caseName + bmpExtension), noisy)
		# calculate cleansed version and write to output folder


		mask = makeButterworthLowPassMask(lowPassD[i], 3, noisy.shape)
		modulo, phase = getModuloAndPhase(noisy)
		modulo = modulo*mask
		result = recombineModuloAndPhase(modulo, phase)
		result, _, _ = unsharpMasking(result, 2, unsharpD[i], 5)

		#result = cv2.medianBlur(noisy, 3)

		cv2.imwrite(os.path.join(caseName, caseName + "_result" + bmpExtension), result)
		# generate report and save it into output folder, print on console
		reportLines = generateReportLines(caseName, face, noisy, None, None, None, result)
		with(open(os.path.join(caseName, caseName + '_report' + txtFileExtesion), 'w+')) as file:		
			file.writelines(reportLines)
			for line in reportLines: print(line)
		# display images
		displayResult([face, result, noisy], ['noisy', 'cleansed', f'{caseName}'])
		

	# Face_thermogram Images - all cases treated with median spatial filter

	# load main face_thermogram image
	face = cv2.imread(os.path.join(grayNoisyImagesSourcePath, "face_thermogram.bmp"))
	face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)	
	# for every noisy face image...
	caseNames = ["face_thermogram_1", "face_thermogram_2", "face_thermogram_3", "face_thermogram_4", "face_thermogram_5"]
	for caseName in caseNames:
		# load noisy image
		noisy = cv2.imread(os.path.join(grayNoisyImagesSourcePath, caseName + bmpExtension))
		noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
		# create output folder for case
		createDirIfNotExists(caseName)
		# write original and noisy image into output folder
		cv2.imwrite(os.path.join(caseName, "face_thermogram" + bmpExtension), face)
		cv2.imwrite(os.path.join(caseName, caseName + bmpExtension), noisy)
		# calculate cleansed version and write to output folder
		result = cv2.medianBlur(noisy, 3)
		cv2.imwrite(os.path.join(caseName, caseName + "_result" + bmpExtension), result)
		# generate report and save it into output folder, print on console
		reportLines = generateReportLines(caseName, face, noisy, None, None, None, result)
		with(open(os.path.join(caseName, caseName + '_report' + txtFileExtesion), 'w+')) as file:		
			file.writelines(reportLines)
			for line in reportLines: print(line)
		# display images
		displayResult([face, result, noisy], ['noisy', 'cleansed', f'{caseName}'])


	# Finger images
	finger = cv2.imread(os.path.join(grayNoisyImagesSourcePath, "finger.bmp"))
	finger = cv2.cvtColor(finger, cv2.COLOR_BGR2GRAY)	

	outputFolder = caseName = 'finger1'
	createDirIfNotExists(outputFolder)
	results = caseFinger1()
	reportLines = writeCaseData(caseName, outputFolder, finger, *results)
	displayCaseImages(caseName, finger, *results)
	for i in reportLines: print(i)
	
	outputFolder = caseName = 'finger2'
	createDirIfNotExists(outputFolder)
	results = caseFinger2()
	reportLines = writeCaseData(caseName, outputFolder, finger, *results)
	displayCaseImages(caseName, finger, *results)
	for i in reportLines: print(i)
	
	outputFolder = caseName = 'finger3'
	createDirIfNotExists(outputFolder)
	results = caseFinger3()
	reportLines = writeCaseData(caseName, outputFolder, finger, *results)
	displayCaseImages(caseName, finger, *results)
	for i in reportLines: print(i)

	outputFolder = caseName = 'finger4'
	createDirIfNotExists(outputFolder)
	results = caseFinger4()
	reportLines = writeCaseData(caseName, outputFolder, finger, *results)
	displayCaseImages(caseName, finger, *results)
	for i in reportLines: print(i)
	
	outputFolder = caseName = 'finger5'
	createDirIfNotExists(outputFolder)
	results = caseFinger5()
	reportLines = writeCaseData(caseName, outputFolder, finger, *results)
	displayCaseImages(caseName, finger, *results)
	for i in reportLines: print(i)
	
	# Iris images
	iris = cv2.imread(os.path.join(grayNoisyImagesSourcePath, "iris.bmp"));
	iris = cv2.cvtColor(iris, cv2.COLOR_BGR2GRAY)

	# For every blurry iris image...
	caseNames = ["iris_1", "iris_2", "iris_3", "iris_4", "iris_5"]
	unsharpK = [1, 1, 1, 1.5, 2, 2]
	butterworthCutoffDistance = [40, 30, 22, 16, 12]
	for i, caseName in enumerate(caseNames):
		# load blurry image
		blurry = cv2.imread(os.path.join(grayNoisyImagesSourcePath, caseName + bmpExtension))
		blurry = cv2.cvtColor(blurry, cv2.COLOR_BGR2GRAY)
		# create output folder for case
		createDirIfNotExists(caseName)
		# calculate improved version and write to output folder
		result, mask, blurred = unsharpMasking(blurry, 1, butterworthCutoffDistance[i], 3)
		cv2.imwrite(os.path.join(caseName, caseName + "_result" + bmpExtension), result)
		cv2.imwrite(os.path.join(caseName, caseName + "_mask" + bmpExtension), getMaskDisplayImage(mask))
		# generate report and save it into output folder, print on console
		
		reportLines = generateReportLines(caseName, iris, blurry, None, None, None, result)
		with(open(os.path.join(caseName, caseName + '_report' + txtFileExtesion), 'w+')) as file:		
			file.writelines(reportLines)
			for line in reportLines: print(line)
		# display images
		displayResult([iris, result, blurry, blurred], ['original', 'improved', f'{caseName}', 'blurred'], 2)

	# Color Thermogram
	face = cv2.imread(os.path.join(rgbNoisyImagesSourcePath, "face.png"));
	
	caseName = "color_thermogram_1"
	# create output folder for case
	createDirIfNotExists(caseName)
	# reading the noisy image
	noisy = cv2.imread(os.path.join(rgbNoisyImagesSourcePath, "face1.png"));
	# get channels (in volor version for viewing)
	channels = cv2.split(noisy)
	blue = cv2.merge([channels[0], np.zeros(channels[0].shape, np.uint8), np.zeros(channels[0].shape, np.uint8)])
	green = cv2.merge([np.zeros(channels[0].shape, np.uint8), channels[1], np.zeros(channels[0].shape, np.uint8)])
	red = cv2.merge([np.zeros(channels[0].shape, np.uint8), np.zeros(channels[0].shape, np.uint8), channels[2]])
	# save channels
	for i, color in enumerate(["blue", "green", "red"]):
		cv2.imwrite(os.path.join(caseName, f'{caseName}_{color}Channel{bmpExtension}'), channels[i])
	# median filter red channel and recombine result
	result = cv2.merge([noisy[:, :, 0], noisy[:, :, 1], cv2.medianBlur(noisy[:, :, 2], 3)])
	# save result
	cv2.imwrite(os.path.join(caseName, f'{caseName}_cleansed{bmpExtension}'), channels[i])
	# write the report
	with(open(os.path.join(caseName, caseName + '_report' + txtFileExtesion), 'w+')) as file:		
		file.writelines(generateReportLines(caseName, face, noisy, None, None, None, result))
	# displaying result along 
	displayResult([face, result, noisy, blue, green, red], ['original', 'improved', f'{caseName}', 'noisyBlue', 'noisyGreen', 'noisyRed'], 2)
	
	caseName = "color thermogram 2"
	# reading the noisy image
	noisy = cv2.imread(os.path.join(rgbNoisyImagesSourcePath, "face2.png"));
	# swapping blue and red channels
	result = cv2.merge([noisy[:, :,2], noisy[:, :, 1], noisy[:, :, 0]])
	# create output folder for case
	createDirIfNotExists(caseName)
	# write the report
	with(open(os.path.join(caseName, caseName + '_report' + txtFileExtesion), 'w+')) as file:		
		file.writelines(generateReportLines(caseName, face, noisy, None, None, None, result))
	# displaying result
	displayResult([face, result, noisy], ['original', 'improved', f'{caseName}'], 2)

	caseName = "color thermogram 3"
	# reading the noisy image
	noisy = cv2.imread(os.path.join(rgbNoisyImagesSourcePath, "face3.png"));
	#hS = cv2.calcHist([noisy[:, :, 1]], [0], None, [256], [0, 256])
	#hF = cv2.calcHist([face[:, :, 1]], [0], None, [256], [0, 256])
	#displayResult([hS, hF], "SF", 1, [0, 1])

	# making a lookup table for enhancing channels B and G
	h = noisy[:, :, 0].max()
	t = [x*255/h if x <= h else 255 for x in range(256)]
	
	# transform pixels
	result = np.array([[[t[noisy[y, x, 0]], t[noisy[y, x, 1]], noisy[y, x, 2]] for x in range(noisy.shape[1])] for y in range(noisy.shape[0])], np.uint8)
	# create output folder for case
	createDirIfNotExists(caseName)
	# write the report
	with(open(os.path.join(caseName, caseName + '_report' + txtFileExtesion), 'w+')) as file:		
		file.writelines(generateReportLines(caseName, face, noisy, None, None, None, result))
	# displaying result
	displayResult([face, result, noisy], ['original', 'improved', f'{caseName}'], 2)		


	caseName = "color thermogram 4"
	# reading the noisy image
	noisy = cv2.imread(os.path.join(rgbNoisyImagesSourcePath, "face4.png"));
	
	compareRgbAndHsiHistograms(face, noisy)
	
	# dont't know what to do with this one

	# displaying result
	#displayResult([face, result, noisy], ['original', 'improved', f'{caseName}'], 2)

	# Lena
	lena = cv2.imread(os.path.join(rgbNoisyImagesSourcePath, "lena.png"));

	caseName = "lena1"
	# reading the noisy image
	noisy = cv2.imread(os.path.join(rgbNoisyImagesSourcePath, "lena1.png"));
	# converting to hsi
	hsi = bgr2hsi(noisy)
	# making a lookup table for enhancin I channel
	h = noisy[:, :, 0].max()
	l = 240
	t = [x*l/h if x <= h else 255 for x in range(256)]
	# applying the transformation
	result = np.array([[[hsi[y, x, 0], hsi[y, x, 1], t[hsi[y, x, 2]]] for x in range(hsi.shape[1])] for y in range(hsi.shape[0])], np.uint8)
	# converting back to BGR
	result = hsi2bgr(result)
	# create output folder for case
	createDirIfNotExists(caseName)
	# write the report
	with(open(os.path.join(caseName, caseName + '_report' + txtFileExtesion), 'w+')) as file:		
		file.writelines(generateReportLines(caseName, lena, noisy, None, None, None, result))
	# desplay results
	displayResult([lena, result, noisy], ['original', 'improved', f'{caseName}'], 2)
	

	caseName = "lena2"
	# reading the noisy image
	noisy = cv2.imread(os.path.join(rgbNoisyImagesSourcePath, "lena2.png"));
	# converting to hsi
	hsi = bgr2hsi(noisy)
	# move Hue by approx. -8.5 degrees
	hsi[:, :, 0] = (hsi[:, :, 0] + 249)%256
	# generate an intensity transform lookup table for saturation and intensity
	h = 255
	l = 128
	ts = [x*l/h if x <= h else 255 for x in range(256)]
	# apply the transform
	result = np.array([[[hsi[y, x, 0], ts[hsi[y, x, 1]], ts[hsi[y, x, 2]]] for x in range(hsi.shape[1])] for y in range(hsi.shape[0])], np.uint8)
	# convert back to bgr
	result = hsi2bgr(hsi)
	# create output folder for case
	createDirIfNotExists(caseName)
	# write the report
	with(open(os.path.join(caseName, caseName + '_report' + txtFileExtesion), 'w+')) as file:		
		file.writelines(generateReportLines(caseName, lena, noisy, None, None, None, result))
	# display results
	displayResult([lena, result, noisy], ['original', 'improved', f'{caseName}'], 2)
	

	caseName = "lena3"
	# reading the noisy image
	noisy = cv2.imread(os.path.join(rgbNoisyImagesSourcePath, "lena3.png"));
	# generating individual lookup tables for each BGR channel
	# lookup table for blue
	h = 100
	l = 163
	tb = [x*l/h if x <= h else 255 for x in range(256)]
	# lookup table for green
	h = 100
	l = 215
	tg = [x*l/h if x <= h else 255 for x in range(256)]
	# lookup table for red
	h = 100
	l = 225
	tr = [x*l/h if x <= h else 255 for x in range(256)]
	# applying the transforms
	bgr = np.array([[[tb[noisy[y, x, 0]], tg[noisy[y, x, 1]], tr[noisy[y, x, 2]]] for x in range(hsi.shape[1])] for y in range(hsi.shape[0])], np.uint8)
	#converting transformed image to HSI
	hsi = bgr2hsi(bgr)
	# make lookup table for adjusting saturation
	# lookup table for red
	h = 100
	l = 75
	ts = [x*l/h if x <= h else 255 for x in range(256)]
	# applying transform to S channel
	S = np.array([[ts[hsi[y, x, 1]] for x in range(hsi.shape[1])] for y in range(hsi.shape[0])], np.uint8)	
	# recombining HSI channels
	result = cv2.merge([hsi[:, :, 0], S, hsi[:, :, 2]]) 
	# converting back to BGR
	result = hsi2bgr(result)
	# create output folder for case
	createDirIfNotExists(caseName)
	# write the report
	with(open(os.path.join(caseName, caseName + '_report' + txtFileExtesion), 'w+')) as file:		
		file.writelines(generateReportLines(caseName, lena, noisy, None, None, None, result))
	# displaying the results
	displayResult([lena, result, noisy], ['original', 'improved', f'{caseName}'], 2)
	
	caseName = "lena4"
	# reading the noisy image
	noisy = cv2.imread(os.path.join(rgbNoisyImagesSourcePath, "lena4.png"));
	channels = cv2.split(noisy)
	displayResult(list(channels), ['original', 'improved', f'{caseName}'], 2)

	# lookup table for blue channel
	h = 255
	l = 100
	t = [x*l/h if x <= h else 255 for x in range(256)]
	# applying transform
	blue = np.array([[t[channels[0][y, x]] for x in range(noisy.shape[1])] for y in range(noisy.shape[0])], np.uint8)	

	# recombine channels
	result = cv2.merge([blue, channels[1], channels[2]])
	# create output folder for case
	createDirIfNotExists(caseName)
	# write the report
	with(open(os.path.join(caseName, caseName + '_report' + txtFileExtesion), 'w+')) as file:		
		file.writelines(generateReportLines(caseName, lena, noisy, None, None, None, result))
	# displaying the results
	displayResult([lena, result, noisy], ['original', 'improved', f'{caseName}'], 2)
	#compareRgbAndHsiHistograms(lena, result)	



if __name__ == "__main__":
	generateImages()