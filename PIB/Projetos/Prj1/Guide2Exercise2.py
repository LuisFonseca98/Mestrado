import cv2
import matplotlib.pyplot as plt

"""
Changes the contrast of an image
@alpha: Contrast control 
        To lower the contrast, use 0 < alpha < 1 
        And for higher contrast use alpha > 1.
        
@beta: brightness control
        A good range for brightness value is [-127, 127]
"""

def changeBrightness(image,pathToSave,SavedHistograms,histogramName,alpha=0.5,beta=10):
    
    imageRead = cv2.imread(image)
    imageAdjusted = cv2.convertScaleAbs(imageRead,alpha,beta)
    
    histg = cv2.calcHist([imageRead],[0],None,[256],[0,256]) 
    newHistgImage = cv2.calcHist([imageAdjusted],[0],None,[256],[0,256]) 
        
    print('Original Image: ', imageRead)
    print('Image Adjusted(Brightness)', imageAdjusted)
    
    cv2.imshow('Original Image', imageRead)
    cv2.imshow('Brightness Image', imageAdjusted)
    
    cv2.imwrite(pathToSave, imageAdjusted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    plt.title('Histogram Original Image')
    plt.plot(histg) 
    plt.show() 
    
    plt.title('Histogram Brightness Image')
    plt.plot(newHistgImage)
    plt.savefig(SavedHistograms + histogramName)
    plt.show() 
    
"""
Changes the contrast of an image
@alpha: Contrast control 
        To lower the contrast, use 0 < alpha < 1 
        And for higher contrast use alpha > 1.
        
@beta: brightness control
        A good range for brightness value is [-127, 127]
        
        
@gama
    
"""  
def changeContrast(image,pathToSave, SavedHistograms,histogramName,alpha=0.25,gamma=-2):
    
    imageRead = cv2.imread(image)
    imageAdjusted = cv2.addWeighted(imageRead, alpha, imageRead, 0, gamma)
    
    #histg = cv2.calcHist([imageRead],[0],None,[256],[0,256]) 
    newHistgImage = cv2.calcHist([imageAdjusted],[0],None,[256],[0,256]) 
        
    print('Original Image: ', imageRead)
    print('Image Adjusted(Contrast)', imageAdjusted)
    
    #cv2.imshow('Original Image', imageRead)
    cv2.imshow('Contrasted Image', imageAdjusted)
    
    cv2.imwrite(pathToSave, imageAdjusted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #plt.title('Histogram Original Image')
    #plt.plot(histg) 
    #plt.show() 
    
    plt.title('Histogram Contrast Image')
    plt.plot(newHistgImage)
    plt.savefig(SavedHistograms + histogramName)
    plt.show() 
    
if __name__ == "__main__":

    image = 'Dataset/face1.jpg'
    image2 = 'Dataset/face2.bmp'
    image3 = 'Dataset/face3.jpg'
    image4 = 'Dataset/face4.jpg'
    image5 = 'Dataset/face5.jpg'
    image6 = 'Dataset/face6.jpg'
    image7 = 'Dataset/face7.jpg'
    imageLena = 'Dataset/Lena.jpg'
    imageGreyLena = 'Dataset/lenagray.png'
    pathToSavePictures = 'SavedPictures/'
    SavedHistograms = 'HistogramsPics/'
    
    changeBrightness(image3,pathToSavePictures + 'New Brightness Image.png',SavedHistograms, 'Histogram Brightness.png')
    changeContrast(image3,pathToSavePictures + 'New Contrast Image.png',SavedHistograms, 'Histogram Contrast.png')
    


    
    