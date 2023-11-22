import cv2
import numpy as np
from PIL import Image



def threshold(image):
    
    th, im_th = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary Image', im_th)
    print('Th: ',th)
    print('im_th: ',im_th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
def binaryImage(image):
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    th, im_gray_th_otsu = cv2.threshold(im_gray, 128, 192, cv2.THRESH_OTSU)
    cv2.imshow('Threshold Image', im_gray_th_otsu)
    print('th: ', th)
    print('im_gray_th_otsu: ', im_gray_th_otsu)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def binarization_for_grayscale_image(image):
    
    im_gray = np.array(Image.open(image).convert('L'))
    print(type(im_gray))

    #first case
    thresh = 128
    #im_bool = im_gray > thresh
    #print(im_bool)

    #second case
    #maxval = 255
    #im_bin = (im_gray > thresh) * maxval
    #print(im_bin)

    #third case
    im_bin_keep = (im_gray > thresh) * im_gray
    print(im_bin_keep)


def build_color_image(image):

    im_gray = np.array(Image.open(image).convert('L'))

    im_bool = im_gray > 128
    im_dst = np.empty((*im_gray.shape, 3))
    r, g, b = 128, 160, 192

    #im_bool = im_gray > 128
    #im_dst = np.empty((*im_gray.shape, 3))
    #r, g, b = 255, 128, 32

    #im_dst[:, :, 0] = im_bool * r #color R component
    #im_dst[:, :, 1] = im_bool * g #color G component
    #im_dst[:, :, 2] = im_bool * b #color B component

    #with negation attribute "~"
    im_dst[:, :, 0] = im_bool * r
    im_dst[:, :, 1] = ~im_bool * g
    im_dst[:, :, 2] = im_bool * b

    Image.fromarray(np.uint8(im_dst)).save('numpy_binarization_color.png')



image = cv2.imread('Lena.jpg')
threshold(image)
binaryImage(image)
#binarization_for_grayscale_image(image)
#binarization_for_grayscale_image(image)
#build_color_image(image)