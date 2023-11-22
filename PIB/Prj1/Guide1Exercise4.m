

function image_information()

%closes all the figures
close all

%cleans the console
clc

%we need to pass the name of the picture
I = imread('Dataset\face1.jpg');
I2 = imread('Dataset\face2.bmp');
I3 = imread('Dataset\face3.jpg');

%save the rows and columns of the image
[M,N] = size(I);
[M2,N2] = size(I2);
[M3,N3] = size(I3);

histI1=imhist(I);
histI2=imhist(I2);
histI3=imhist(I3);

DistinctPixelImage=length(find(histI1>0));
DistinctPixelImage2=length(find(histI2>0));
DistinctPixelImage3=length(find(histI3>0));

fprintf('Pixeis Distintos Image 1=%d\n', DistinctPixelImage);
fprintf('Pixeis Distintos Image 2=%d\n', DistinctPixelImage2);
fprintf('Pixeis Distintos Image 3=%d\n', DistinctPixelImage3);
fprintf(' ')

%calculate the entropy for the different images
H = entropy(I);
H2 = entropy(I2);
H3 = entropy(I3);

%calculate the avg intensity
m = sum(sum(I)) / (M * N);
m2 = sum(sum(I2)) / (M2 * N2);
m3 = sum(sum(I3)) / (M3 * N3);

%prints on the console the different values
fprintf('Entropy=%d\n, Intensity=%d\n',H,m);
fprintf('Entropy2=%d\n, Intensity2=%d\n',H2,m2);
fprintf('Entropy3=%d\n, Intensity3=%d\n',H3,m3);

%to compute the negative value of an image
L = 2 ^ 8;
negativeImage = (L - 1) - I;
negativeImage2 = (L - 1 ) - I2
negativeImage3 = (L - 1 ) - I3

%shows the results in plots
figure(1)
subplot(121); imshow(I); colorbar; title('Image');
subplot(122); imhist(I); title( sprintf(' Histogram. H=%.2f\n',H) );
impixelinfo

% figure(2)
subplot(212); imshow(negativeImage); colorbar; title('Negative Image');
subplot(222); imhist(negativeImage); title(sprintf('Negative Image Histogram'));
impixelinfo

figure(3)
subplot(313); imshow(negativeImage2); colorbar; title('Negative Image2');
subplot(312); imhist(negativeImage2); title(sprintf('Negative Image 2 Histogram'));
impixelinfo

figure(5)
subplot(313); imshow(negativeImage3); colorbar; title('Negative Image3');
subplot(312); imhist(negativeImage3); title(sprintf('Negative Image 3 Histogram'));
impixelinfo

figure(6)
subplot(313); imshow(I2); colorbar; title("Image2");
subplot(312); imhist(I2); title( sprintf(' HistogramI2. H2=%.2f\n',H2) );
impixelinfo

figure(7)
subplot(411); imshow(I3); colorbar; title("Image3");
subplot(412); imhist(I3); title( sprintf(' HistogramI3. H3=%.2f\n',H3) );
impixelinfo

end





