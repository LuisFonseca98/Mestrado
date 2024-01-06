import cv2
import os


def fingerPrintRecognition(samplePath, resultName):
    samples = cv2.imread(samplePath)
    filename = None
    image = None
    best_score = 0
    kp1, kp2, mP = None, None, None
    counter = 0
    for file in [file for file in os.listdir('Dataset/FingerprintDataset')][:1000]:

        if counter % 10 == 0:
            print(counter)
            print(file)

            counter += 1
        fingerprint_image = cv2.imread('Dataset/FingerprintDataset/' + file)
        sift = cv2.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(samples, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {}).knnMatch(descriptors_1, descriptors_2, k=2)

        match_points = []
        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                match_points.append(p)

        keypoints = 0
        if len(keypoints_1) < len(keypoints_2):
            keypoints = len(keypoints_1)
        else:
            keypoints = len(keypoints_2)

        if len(match_points) / keypoints * 100 > best_score:
            best_score = len(match_points)
            filename = file
            image = fingerprint_image
            kp1, kp2, mP = keypoints_1, keypoints_2, match_points

    print("BEST SCORE: ", filename)
    print("SCORE: " + str(best_score))

    result = cv2.drawMatches(samples, kp1, image, kp2, mP, None)
    result = cv2.resize(result, None, fx=0.5, fy=0.5)

    cv2.imshow('Result', result)
    cv2.imwrite('SavedPictures/FingerprintRecognition/' + resultName + '.jpg', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result


if __name__ == "__main__":
    samplePath_LittleOptica = 'Dataset/FingerprintDatasetAFTER/LittleOptica_AfterBrightness.bmp'
    samplePath_IndexCapacitive = 'Dataset/FingerprintDatasetAFTER/IndexCapacitive_AfterBrightness.bmp'
    samplePath_MediumCapacitive = 'Dataset/FingerprintDatasetAFTER/MediumCapacitive_AfterBrightness.bmp'
    samplePath_RingOptical = 'Dataset/FingerprintDatasetAFTER/RingOptical_AfterBrightness.bmp'
    samplePath_ThumbCapacitive = 'Dataset/FingerprintDatasetAFTER/ThumbCapacitive_AfterBrightness.bmp'

    fingerPrintRecognition(samplePath_LittleOptica, 'Fingerprint_recognition_LittleOptica')
    #fingerPrintRecognition(samplePath_IndexCapacitive, 'Fingerprint_recognition_IndexCapacitive')
    #fingerPrintRecognition(samplePath_MediumCapacitive, 'Fingerprint_recognition_MediumCapacitive')
    #fingerPrintRecognition(samplePath_RingOptical, 'Fingerprint_recognition_RingOptical')
    #fingerPrintRecognition(samplePath_ThumbCapacitive, 'Fingerprint_recognition_ThumbCapacitive')
