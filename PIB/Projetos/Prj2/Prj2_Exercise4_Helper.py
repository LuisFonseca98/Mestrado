import cv2
import matplotlib.pyplot as plt
import os
import face_recognition as fr

# Filesystem paths
baseSavePath = os.path.join(os.getcwd(), 'SavedPictures')
featureDetectionImagesPath = os.path.join(baseSavePath, 'FaceDetection')
featureRecognitionImagesPath = os.path.join(baseSavePath, 'FaceRecognition')

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
    cv2.imwrite(os.path.join(path, fileName), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.imwrite(os.path.join(path, fileName), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


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


def faceDetectionImages(imagePath):

    image_read = cv2.imread(imagePath)
    _, fileNameExt = os.path.split(imagePath)

    fileName, _ = os.path.splitext(fileNameExt)

    # Setting up face classifier. Load pretrained cascade Haar classifier from file
    faceClassifier = cv2.CascadeClassifier('Classifiers/haarcascade_frontalface_default.xml')

    # Confidence threshold for face classifier results: detections below this value get rejected
    # This value was obtained experimentally
    faceConfidenceThreshold = 0.0

    # Convert to grayscale for applying the classifier
    gray_image = cv2.cvtColor(image_read, cv2.COLOR_BGR2GRAY)

    # Get a set of rectangles around faces in the image
    faces, _, confidenceLevels = faceClassifier.detectMultiScale3(gray_image, outputRejectLevels=True)

    # For every face detected
    for i, (x, y, w, h) in enumerate(faces):
        if confidenceLevels[i] > faceConfidenceThreshold:

            # Draw a rectangle around the face
            cv2.rectangle(image_read, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Save the result
    # displaying and saving results
    displayResult([image_read], ['Face Detection'])
    saveImageInJpgAndPngFormats(image_read, 'Face Detection Feature ' + fileNameExt, featureDetectionImagesPath)

    return featureDetectionImagesPath


def process_images(folder_pictures, target_image_path):

    def encode_faces(folder_pictures):
        list_people_encoded = []

        for filename in os.listdir(folder_pictures):
            known_image = fr.load_image_file(f'{folder_pictures}/{filename}')
            known_encoding = fr.face_encodings(known_image)[0]

            list_people_encoded.append((known_encoding, filename))

        return list_people_encoded

    def create_frame(location, label):
        top, right, bottom, left = location
        cv2.rectangle(target_image, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.rectangle(target_image, (left, bottom + 20), (right, bottom), (255, 0, 0), cv2.FILLED)
        cv2.putText(target_image, label, (left + 3, bottom + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    target_image = fr.load_image_file(target_image_path)
    target_encoding = fr.face_encodings(target_image)[0]
    face_location = fr.face_locations(target_image)
    _, fileNameExt = os.path.split(target_image_path)

    for person in encode_faces(folder_pictures):
        encoded_face = person[0]
        filename = person[1]

        is_target_face = fr.compare_faces([encoded_face], target_encoding, tolerance=0.55)
        print(f'{is_target_face} {filename}')

        if face_location:
            face_number = 0
            for location in face_location:
                if is_target_face[face_number]:
                    label = filename
                    create_frame(location, label)
                face_number += 1

    displayResult([target_image], ['Face Recognition'])
    saveImageInJpgAndPngFormats(target_image, 'Face Recognition Feature ' + fileNameExt,featureRecognitionImagesPath)

    return target_image


if __name__ == "__main__":

    #####DATASET#########
    image = 'Dataset/OriginalDataset/face1.jpg'
    image2 = 'Dataset/OriginalDataset/face2.bmp'
    image3 = 'Dataset/OriginalDataset/face3.jpg'
    image4 = 'Dataset/OriginalDataset/face4.jpg'
    image5 = 'Dataset/OriginalDataset/face5.jpg'
    image6 = 'Dataset/OriginalDataset/face6.jpg'
    image7 = 'Dataset/OriginalDataset/face7.jpg'
    imageLena = 'Dataset/OriginalDataset/Lena.jpg'

    #faceDetectionImages(image)
    process_images('Dataset/FaceRecognitionDataset',image)

    #faceDetectionImages(image3)
    process_images('Dataset/FaceRecognitionDataset',image3)

    #faceDetectionImages(image5)
    process_images('Dataset/FaceRecognitionDataset',image5)

    #faceDetectionImages(image6)
    process_images('Dataset/FaceRecognitionDataset',image6)

    #faceDetectionImages(image2)
    process_images('Dataset/FaceRecognitionDataset',image7)



