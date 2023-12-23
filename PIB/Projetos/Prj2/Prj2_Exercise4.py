import cv2
import os
import matplotlib.pyplot as plt

baseSavePath = os.path.join(os.getcwd(), 'SavedPictures')
featureDetectionImagesPath = os.path.join(baseSavePath, 'FeatureDetection')
# File extensions
jpegExtension = '.jpg'
pngExtension = '.png'

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
    faces, _, confidenceLevels = faceClassifier.detectMultiScale3(grayImage, outputRejectLevels=True)

    # for every face detected
    for i, (x, y, w, h) in enumerate(faces):
        if confidenceLevels[i] > faceConfidenceThreshold:
            # extracting face area
            roi = grayImage[y:y + h, x:x + w]

            # searching for eyes within face area
            eyes, _, levels = eyeClassifier.detectMultiScale3(roi, minNeighbors=1, outputRejectLevels=True)
            for j, (ex, ey, ew, eh) in enumerate(eyes):
                if (levels[j] > eyeConfidenceThreshold):
                    cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

            # searching for mouths within face area
            mouths, _, levels = mouthClassifier.detectMultiScale3(roi, minNeighbors=3, outputRejectLevels=True)
            for j, (mx, my, mw, mh) in enumerate(mouths):
                if (levels[j] > mouthConfidenceThreshold):
                    cv2.rectangle(img, (x + mx, y + my), (x + mx + mw, y + my + mh), (0, 0, 255), 2)

    # displaying and saving results
    displayResult([img], ['eyes and mouth detection'])
    saveImageInJpgAndPngFormats(img, fileNameExt, featureDetectionImagesPath)

    return featureDetectionImagesPath


def face_recognition(imagePath):

    img = cv2.imread(imagePath)
    _, fileNameExt = os.path.split(imagePath)
    fileName, _ = os.path.splitext(fileNameExt)


def displayResult(displayData, titles, nRows=1, plotIndices=[]):
    # display images with matplotlib
    nCols = int(len(displayData) / nRows)
    for i, image in enumerate(displayData):
        plt.subplot(nRows, nCols, i + 1)
        plt.title(titles[i])
        if i in plotIndices:
            plt.plot(image)
        else:
            plt.imshow(cv2.cvtColor(displayData[i], cv2.COLOR_BGR2RGB))
            plt.xticks([])
            plt.yticks([])

    plt.show()


def saveImageInJpgAndPngFormats(image, fileName, path):
    cv2.imwrite(os.path.join(path, fileName + jpegExtension), image)
    cv2.imwrite(os.path.join(path, fileName + pngExtension), image)
