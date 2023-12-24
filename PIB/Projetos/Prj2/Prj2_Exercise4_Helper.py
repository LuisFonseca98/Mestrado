import cv2
import matplotlib.pyplot as plt
import os
import face_recognition as fr

# Filesystem paths
baseSavePath = os.path.join(os.getcwd(), 'SavedPictures')
featureDetectionImagesPath = os.path.join(baseSavePath, 'FaceDetection')
featureRecognitionImagesPath = os.path.join(baseSavePath, 'FaceRecognition')


# File extensions
jpegExtension = '.jpg'
pngExtension = '.png'

featureDetectedFilenameSuffix = '_faceDetection'
featureRecognitionFilenameSuffix = '_faceRecognition'

"""
Making sure required destination directories for results exist
"""
def setupFilesystem():
    # base path for destination directories
    if not os.path.exists(baseSavePath):
        os.mkdir(baseSavePath)

    # destination directories for DIP operation results
    for path in [featureRecognitionImagesPath, featureDetectionImagesPath]:
        if not os.path.exists(path):
            os.mkdir(path)


"""
Saving cv image in JPEG and PNG formats
"""


def saveImageInJpgAndPngFormats(image, fileName, path):
    cv2.imwrite(os.path.join(path, fileName + jpegExtension), image)
    cv2.imwrite(os.path.join(path, fileName + pngExtension), image)


"""
Displaying results using Maptplotlib
Contents of 'displayData' are taken as image data except for entries which indices are contained in 'plotIndices'
in those cases, the entries are taken as plot data
"""
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


def encode_faces(folder):
    list_people_encoding = []

    for filename in os.listdir(folder):
        know_image = fr.load_image_file(f'{folder}/{filename}')
        know_encoding = fr.face_encodings(know_image)[0]

        list_people_encoding.append((know_encoding, filename))

    return list_people_encoding


def find_target_face():
    face_location = fr.face_locations(target_image)

    for person in encode_faces('Dataset/FaceRecognitionDataset'):
        encoded_face = person[0]
        filename = person[1]

        is_target_face = fr.compare_faces(encoded_face, target_encoding, tolerance=0.55)
        print(f'{is_target_face} {filename}')

        if face_location:
            face_number = 0
            for location in face_location:
                if is_target_face[face_number]:
                    label = filename
                    create_frame(location, label)
                face_number += 1


def create_frame(location, label):
    top, right, bottom, left = location
    cv2.rectangle(target_image, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.rectangle(target_image, (left, bottom + 20), (right, bottom), (255, 0, 0), cv2.FILLED)
    cv2.putText(target_image, label, (left + 3, bottom + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def render_image(imagePath):
    rgb_img = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    _, fileNameExt = os.path.split(imagePath)

    cv2.imshow('Face Recognition', rgb_img)
    cv2.waitKey(0)

    displayResult([rgb_img], ['Face Recognition'])
    saveImageInJpgAndPngFormats(rgb_img, 'Face Recognition Result' + fileNameExt, featureRecognitionImagesPath)

def faceDetectionImages(imagePath):

    img = cv2.imread(imagePath)
    _, fileNameExt = os.path.split(imagePath)

    fileName, _ = os.path.splitext(fileNameExt)

    # Setting up face classifier. Load pretrained cascade Haar classifier from file
    faceClassifier = cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')

    # Confidence threshold for face classifier results: detections below this value get rejected
    # This value was obtained experimentally
    faceConfidenceThreshold = 0.0

    # Convert to grayscale for applying the classifier
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get a set of rectangles around faces in the image
    faces, _, confidenceLevels = faceClassifier.detectMultiScale3(grayImage, outputRejectLevels=True)

    # For every face detected
    for i, (x, y, w, h) in enumerate(faces):
        if confidenceLevels[i] > faceConfidenceThreshold:
            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Displaying and saving results
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    # displaying and saving results
    displayResult([img], ['Face Detection'])
    saveImageInJpgAndPngFormats(img, fileNameExt, featureDetectionImagesPath)

    return featureDetectionImagesPath

if __name__ == "__main__":

    #####DATASET#########
    image = 'Dataset/FaceRecognitionDataset/alberto.jpg'
    image2 = 'Dataset/FaceRecognitionDataset/carlos.jpg'
    image3 = 'Dataset/FaceRecognitionDataset/diogo.jpg'
    imageLena = 'Dataset/FaceRecognitionDataset/Lena.jpg'
    image5 = 'Dataset/FaceRecognitionDataset/luis.jpg'
    image6 = 'Dataset/FaceRecognitionDataset/marco.bmp'
    image7 = 'Dataset/FaceRecognitionDataset/mari.jpg'
    image8 = 'Dataset/FaceRecognitionDataset/pedro.jpg'

    result = faceDetectionImages(image)

    print('Result saved at:', result)

    target_image = fr.load_image_file(image)
    target_encoding = fr.face_encodings(target_image)

    find_target_face()
    render_image('FaceRecognition/')
