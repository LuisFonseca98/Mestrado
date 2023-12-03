import numpy as np
import cv2
from matplotlib import pyplot as plt

from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms

def histogramEqualization(image):
    
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    equ = cv2.equalizeHist(img)
    
    plt.plot(cdf_normalized, color = 'b')
    plt.title('Histogram Original Image')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
    
    
    plt.plot(cdf_normalized, color = 'b')
    plt.title('Histogram After Constrast')
    plt.hist(equ.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

    
    cv2.imshow('original image', img)
    cv2.imshow('equ image', equ)
    
    #cv2.imwrite('HistogramsPics/Histogram After Equalization.png')
    cv2.imwrite('SavedPictures/Image After HS.png',equ)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def histogramSpecification():
    
    reference = data.coffee()
    image = data.chelsea()
    
    matched = match_histograms(image, reference, channel_axis=-1)
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()
    
    ax1.imshow(image)
    ax1.set_title('Source')
    ax2.imshow(reference)
    ax2.set_title('Reference')
    ax3.imshow(matched)
    ax3.set_title('Matched')
    
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 8))


    for i, img in enumerate((image, reference, matched)):
        for c, c_color in enumerate(('red', 'green', 'blue')):
            img_hist, bins = exposure.histogram(img[..., c], source_range='dtype')
            axes[c, i].plot(bins, img_hist / img_hist.max())
            img_cdf, bins = exposure.cumulative_distribution(img[..., c])
            axes[c, i].plot(bins, img_cdf)
            axes[c, 0].set_ylabel(c_color)
    
    axes[0, 0].set_title('Source')
    axes[0, 1].set_title('Reference')
    axes[0, 2].set_title('Matched')
        
    plt.tight_layout()
    plt.show()
    

imageLena = 'Dataset/Lena.jpg'
imageFlower = 'Dataset/flower.jpg'
imageSunset = 'Dataset/Sunset.jpg'

histogramEqualization(imageFlower)
histogramSpecification()