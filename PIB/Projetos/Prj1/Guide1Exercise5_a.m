
%applys blur effect to the persons image
function identity_hiding()

%closes all the figures
close all

%cleans the console
clc

image = imread('Dataset\face1.jpg');
Iblur1 = imgaussfilt(image,8);

%blur the image
%H = fspecial('average',50);
%blurImage = imfilter(image,H,'replicate');

%fprintf(image);
%forintf(blurImage);

figure(1)
subplot(121); imshow(image),title('Original Image')
subplot(122); imshow(Iblur1),title('Blur Image')

end







