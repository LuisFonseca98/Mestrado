import numpy as np
import cv2


def and_operation(image1,image2,savedPicture):
    resultAND = cv2.bitwise_and(image1, image2)
    cv2.imshow('Result_AND',resultAND)
    cv2.imwrite('SavedPictures/' + savedPicture, resultAND)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def or_operation(image1,image2,savedPicture):
    resultOR = cv2.bitwise_or(image1, image2)
    cv2.imshow('Result_OR',resultOR)
    cv2.imwrite('SavedPictures/' + savedPicture, resultOR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
image1 = cv2.imread('Dataset/face1.jpg')
image2 = cv2.imread('Dataset/face2.bmp')
image3 = cv2.imread('Dataset/face3.jpg')
image4 = cv2.imread('Dataset/face4.jpg')
image5 = cv2.imread('Dataset/face5.jpg')
image6 = cv2.imread('Dataset/face6.jpg')
image7 = cv2.imread('Dataset/face7.jpg')


rectangle = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)

circle = np.zeros((300, 300), dtype = "uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)

and_operation(rectangle, circle, 'AND Image Test.png')
and_operation(image4, image5,'AND Result 1.png')
and_operation(image6, image7, 'AND Result 2.png')
and_operation(image5, image7, 'AND Result 3.png')

or_operation(rectangle, circle, 'OR Image Test.png')
or_operation(image4, image5, 'OR Result 1.png')
or_operation(image6, image7, 'OR Result 2.png')
or_operation(image5, image7, 'OR Result 3.png')

