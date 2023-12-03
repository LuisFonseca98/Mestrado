import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def threshold(image,pathToSave,SavedHistograms,histogramName):
    
    imageRead = cv2.imread(image)
    th, im_th = cv2.threshold(imageRead, 128, 255, cv2.THRESH_BINARY)    
    
    histg = cv2.calcHist([imageRead],[0],None,[256],[0,256]) 
    newHistgImage = cv2.calcHist([im_th],[0],None,[256],[0,256]) 
    
    print('Th: ',th)
    print('im_th: ',im_th)
    
    cv2.imshow('Original Image', imageRead)
    cv2.imshow('Threshold Result', im_th)
    cv2.imwrite(pathToSave, im_th)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    plt.title('Histogram Original Image')
    plt.plot(histg) 
    plt.show() 
    
    plt.title('Histogram Threshold Image')
    plt.plot(newHistgImage)
    plt.savefig(SavedHistograms + histogramName)
    plt.show() 
 
def binaryImage(image,pathToSave,SavedHistograms,histogramName):
    
    imageRead = cv2.imread(image)
    im_gray = cv2.cvtColor(imageRead, cv2.COLOR_BGR2GRAY)
    th, im_gray_th_otsu = cv2.threshold(im_gray, 128, 255, cv2.THRESH_OTSU)
    
    newHistgImage = cv2.calcHist([im_gray_th_otsu],[0],None,[256],[0,256]) 
    
    cv2.imshow('Binary Result', im_gray_th_otsu)
    cv2.imwrite(pathToSave, im_gray_th_otsu)
    
    print('im_gray_th_otsu: ', im_gray_th_otsu)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    plt.title('Histogram Binary Image')
    plt.plot(newHistgImage)
    plt.savefig(SavedHistograms + histogramName)
    plt.show() 

"""def binarization_for_grayscale_image(image):
    
    im_gray = np.array(Image.open(image).convert('L'))
    print(type(im_gray))

    #first case
    thresh = 128

    #third case
    im_bin_keep = (im_gray > thresh) * im_gray
    print(im_bin_keep)
"""

"""def build_color_image(image):

    im_gray = np.array(Image.open(image).convert('L'))

    im_bool = im_gray > 128
    im_dst = np.empty((*im_gray.shape, 3))
    r, g, b = 128, 160, 192

    #with negation attribute "~"
    im_dst[:, :, 0] = im_bool * r
    im_dst[:, :, 1] = ~im_bool * g
    im_dst[:, :, 2] = im_bool * b

    Image.fromarray(np.uint8(im_dst)).save('numpy_binarization_color.png')
"""

if __name__ == "__main__":
    
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
    SavedHistograms = 'HistogramsPics/'
    
    threshold(imageLena,pathToSavePictures + 'Threshold_FirstImage.png', SavedHistograms, 'Histogram Threshold Image 1.png')
    binaryImage(imageLena,pathToSavePictures + 'Binary_FirstImage.png', SavedHistograms, 'Histogram Binary Image 1.png')
    
    threshold(image2,pathToSavePictures + 'Threshold_SecondImage.png',SavedHistograms, 'Histogram Threshold Image 2.png')
    binaryImage(image2,pathToSavePictures + 'Binary_SecondImage.png',SavedHistograms, 'Histogram Binary Image 2.png')
    
    threshold(image3,pathToSavePictures + 'Threshold_ThirdImage.png',SavedHistograms, 'Histogram Threshold Image 3.png')
    binaryImage(image3,pathToSavePictures + 'Binary_ThirdImage.png',SavedHistograms, 'Histogram Binary Image 3.png')
