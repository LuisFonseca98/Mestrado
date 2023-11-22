
%receives an image
%convert into a grayscale image
%performs face detection
%applys visual effects
function face_detection_effects()

image = imread('Demos\Demo2 - Reading and displaying images and histograms\test.jpg');
imageHat = imread("Demos\Demo2 - Reading and displaying images and histograms\pnghat.png");
imageGray = rgb2gray(image);
%closes all the figures
close all 

%cleans the console
clc

detectorFace = vision.CascadeObjectDetector('LeftEye','MergeThreshold',60);
boundingBox = step(detectorFace,imageGray);
detPic = insertObjectAnnotation(image,'Rectangle',boundingBox,'Eye');


figure

axes('Position',[0.1 0.1 0.7 0.7]);
imshow(detPic);
axes('Position',[0.1 0.6 0.7 0.2]);
imshow(imageHat);


end

