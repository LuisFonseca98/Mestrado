import cv2
import os


def finger_print_recognition(samplePath, resultName):

    samples = cv2.imread(samplePath, cv2.IMREAD_GRAYSCALE)
    filename = None
    image = None
    best_score_for_finger_print = 0
    kp1, kp2, mP = None, None, None
    counter = 0
    for file in [file for file in os.listdir('Dataset/FingerprintDataset')][:100]:

        if counter % 10 == 0:
            print(counter)
            print(file)

            counter += 1
        fingerprint_image = cv2.imread('Dataset/FingerprintDataset/' + file)
        sift = cv2.SIFT_create()

        key_points_1, descriptors_1 = sift.detectAndCompute(samples, None)
        key_points_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        matches = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {}).knnMatch(descriptors_1, descriptors_2, k=2)

        match_points = []
        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                match_points.append(p)

        keypoints = 0
        if len(key_points_1) < len(key_points_2):
            keypoints = len(key_points_1)
        else:
            keypoints = len(key_points_1)

        if len(match_points) / keypoints * 100 > best_score_for_finger_print:
            best_score_for_finger_print = len(match_points)
            filename = file
            image = fingerprint_image
            kp1, kp2, mP = key_points_1, key_points_2, match_points

    print("BEST SCORE: ", filename)
    print("Best_Score_For_Finger_Print: " + str(best_score_for_finger_print))

    result = cv2.drawMatches(samples, kp1, image, kp2, mP, None)
    result = cv2.resize(result, None, fx=0.5, fy=0.5)

    cv2.imshow('Result', result)
    cv2.imwrite('SavedPictures/FingerprintRecognition/' + resultName + '.jpg', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result


def debug_picture(image):
    image_read = cv2.imread(image)

    cv2.imshow('teste', image_read)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    samplePath_LittleOptica_Brightness = 'Dataset/FingerprintDatasetAFTER/LittleOptica_AfterBrightness.bmp'
    samplePath_LittleOptica_Contrast = 'Dataset/FingerprintDatasetAFTER/LittleOptica_AfterContrast.bmp'

    samplePath_IndexCapacitive_Brightness = 'Dataset/FingerprintDatasetAFTER/IndexCapacitive_AfterBrightness.bmp'
    samplePath_IndexCapacitive_Contrast = 'Dataset/FingerprintDatasetAFTER/IndexCapacitive_AfterContrast.bmp'

    samplePath_RingOptical_Brightness = 'Dataset/FingerprintDatasetAFTER/RingOptical_AfterBrightness.bmp'
    samplePath_RingOptical_Contrast = 'Dataset/FingerprintDatasetAFTER/RingOptical_AfterContrast.bmp'

    finger_print_recognition(samplePath_LittleOptica_Brightness, 'Fingerprint_recognition_LittleOptica_Brightness.bmp')
    finger_print_recognition(samplePath_LittleOptica_Contrast, 'Fingerprint_recognition_LittleOptica_Contrast.bmp')

    finger_print_recognition(samplePath_IndexCapacitive_Brightness,
                           'Fingerprint_recognition_IndexCapacitive_Brightness.bmp')
    finger_print_recognition(samplePath_IndexCapacitive_Contrast, 'Fingerprint_recognition_IndexCapacitive_Contrast.bmp')

    finger_print_recognition(samplePath_RingOptical_Brightness, 'Fingerprint_recognition_RingOptical_Brightness.bpm')
    finger_print_recognition(samplePath_RingOptical_Contrast, 'Fingerprint_recognition_RingOptical_Contrast.bpm')
