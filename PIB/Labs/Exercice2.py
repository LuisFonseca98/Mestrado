# import the necessary packages
import numpy as np
import cv2

#FALTA TESTAR COM MAIS IMAGENS!!


def and_operation(image1,image2):
    resultAND = cv2.bitwise_and(image1, image2)
    cv2.imshow("Result_AND", resultAND)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def or_operation(image1,image2):
    resultOR = cv2.bitwise_or(image1, image2)
    cv2.imshow('Result_OR',resultOR)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
image1 = cv2.imread('face4.jpg')
image2 = cv2.imread('face5.jpg')
image3 = cv2.imread('face6.jpg')

rectangle = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)

circle = np.zeros((300, 300), dtype = "uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)


and_operation(image1, image2)
or_operation(image1, image2)


