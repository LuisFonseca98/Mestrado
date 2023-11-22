from Guide2Exercise2 import changeBrightness,changeContrast
from Guide3Exercise2 import threshold,binaryImage

if __name__ == "__main__":
    
    #####DATASET#########

    #para alinea b)
    indexCapacitive = 'Dataset/index_capacitive.bmp'
    mediumCapacitive = 'Dataset/medium_capacitive.bmp'
    thumbCapacitive = 'Dataset/thumb_capacitive.bmp'
    littleOptical = 'Dataset/little_optical.bmp'
    ringOptical = 'Dataset/ring_optical.bmp'
    thumbOptical = 'Dataset/thumb_optical.bmp'
    pathToSavePictures = 'SavedPictures/'
    SavedHistograms = 'HistogramsPics/'
    
    # para alinea c)
    Gray_IndexCapacitive = 'SavedPictures/IndexCapacitive_AfterContrast.bmp'
    Gray_LittleOptical = 'SavedPictures/LittleOptica_AfterContrast.bmp'
    Gray_MediumCapacitive = 'SavedPictures/MediumCapacitive_AfterContrast.bmp'
    Gray_RingOptical = 'SavedPictures/RingOptical_AfterContrast.bmp'
    Gray_ThumbCapacitive = 'SavedPictures/ThumbCapacitive_AfterContrast.bmp'
    Gray_ThumbOptical = 'SavedPictures/ThumbOptical_AfterContrast.bmp'
    
    changeContrast(indexCapacitive,pathToSavePictures + 'IndexCapacitive_AfterContrast.bmp',SavedHistograms, 'Histogram indexCapacitive.png',0.25,15)
    changeContrast(mediumCapacitive,pathToSavePictures + 'MediumCapacitive_AfterContrast.bmp',SavedHistograms, 'Histogram MediumCapacitive.png',0.25,15)
    changeContrast(thumbCapacitive,pathToSavePictures + 'ThumbCapacitive_AfterContrast.bmp',SavedHistograms, 'Histogram LittleOptical.png',0.25,15)
    changeContrast(littleOptical,pathToSavePictures + 'LittleOptica_AfterContrast.bmp',SavedHistograms, 'Histogram MediumCapacitive.png',0.25,50)
    changeContrast(ringOptical,pathToSavePictures + 'RingOptical_AfterContrast.bmp',SavedHistograms, 'Histogram RingOptical.png',0.25,50)
    changeContrast(thumbOptical,pathToSavePictures + 'ThumbOptical_AfterContrast.bmp',SavedHistograms, 'Histogram ThumbOptical.png',0.25,50)
    
    changeBrightness(indexCapacitive,pathToSavePictures + 'IndexCapacitive_AfterBrightness.bmp',SavedHistograms, 'Histogram indexCapacitive.png',beta=3)
    changeBrightness(mediumCapacitive,pathToSavePictures + 'MediumCapacitive_AfterBrightness.bmp',SavedHistograms, 'Histogram MediumCapacitive.png',beta=2)
    changeBrightness(thumbCapacitive,pathToSavePictures + 'ThumbCapacitive_AfterBrightness.bmp',SavedHistograms, 'Histogram LittleOptical.png',beta=3)
    changeBrightness(littleOptical,pathToSavePictures + 'LittleOptica_AfterBrightness.bmp',SavedHistograms, 'Histogram MediumCapacitive.png')
    changeBrightness(ringOptical,pathToSavePictures + 'RingOptical_AfterBrightness.bmp',SavedHistograms, 'Histogram RingOptical.png')
    changeBrightness(thumbOptical,pathToSavePictures + 'ThumbOptical_AfterBrightness.bmp',SavedHistograms, 'Histogram ThumbOptical.png')
    
    binaryImage(Gray_IndexCapacitive,pathToSavePictures + 'IndexCapacitive_Binary.png',SavedHistograms, 'Histogram indexCapacitive.png')
    binaryImage(Gray_MediumCapacitive,pathToSavePictures + 'MediumCapacitive_Binary.png',SavedHistograms, 'Histogram MediumCapacitive.png')
    binaryImage(Gray_ThumbCapacitive,pathToSavePictures + 'ThumbCapacitive_Binary.png',SavedHistograms, 'Histogram LittleOptical.png')
    binaryImage(Gray_LittleOptical,pathToSavePictures + 'LittleOptical_Binary.png',SavedHistograms, 'Histogram MediumCapacitive.png')
    binaryImage(Gray_RingOptical,pathToSavePictures + 'RingOptical_Binary.png',SavedHistograms, 'Histogram RingOptical.png')
    binaryImage(Gray_ThumbOptical,pathToSavePictures + 'IndexCapacitive_ThumbOptical.png',SavedHistograms, 'Histogram ThumbOptical.png')

    
    
    
    
    
    
    

    



    


    