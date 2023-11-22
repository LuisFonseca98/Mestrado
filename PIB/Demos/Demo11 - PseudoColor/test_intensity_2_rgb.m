%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%      
% PIB - Processamento de Imagem e Biometria.
% 
% test_intensity_2_rgb.m
% Analysis of the intensity to RGB transformation technique.

function test_intensity_2_rgb()

% Clear console.
clc

% Close all windows
close all
 
% Input image
%f = imread('squares.gif');
%f = imread('bird.gif');
%f = imread('circuit.tif');
%f = imread('weather.tif');
%f = imread('xray.tif');
f = imread('phantom.tif');
f = imadjust(f);

% Image Resolution
[M,N] = size(f);

% The LUT for the intensity transformation functions for R, G, and B
n  = 0:1:255; 

% Bad choice
TR = abs(cos((pi/115)*n));
TG = abs(cos((pi/110)*n));
TB = abs(cos((pi/105)*n));


% % Good choice
% TR = abs(cos((pi/100)*n));
% TG = abs(cos((pi/100)*n+pi/2));
% TB = abs(cos((pi/100)*n-pi/4));

% Good choice
% TR = abs(cos((pi/150)*n));
% TG = abs(cos((pi/150)*n+pi/2));
% TB = abs(cos((pi/150)*n-pi/4));

% Plot the functions
figure(1);
subplot(311); plot(n,TR); grid on; title('fR');
subplot(312); plot(n,TG); grid on; title('fG');
subplot(313); plot(n,TB); grid on; title('fB');

% Apply the LUT for each function
% Intensity to RGB transform 
fR = intlut(f,uint8(TR*255));
fG = intlut(f,uint8(TG*255));
fB = intlut(f,uint8(TB*255));

% Compose the RGB image
fRGB = uint8(zeros(M,N,3));
fRGB(:,:,1) = fR;
fRGB(:,:,2) = fG;
fRGB(:,:,3) = fB;

% Display both images
figure(2);
ax1 = subplot(121); imagesc(f); 
axis off; title('gray'); colormap(ax1,'gray');
subplot(122); imagesc(fRGB); 
axis off; title('RGB'); colorbar;
impixelinfo 

% Display the images
figure(3);
ax1 = subplot(221); imagesc(f); 
axis off; title('gray'); colormap(ax1,'gray');
ax2 = subplot(222); imagesc(fR); 
axis off; title('R'); colormap(ax2,'gray');
ax3 = subplot(223); imagesc(fG); 
axis off; title('G'); colormap(ax3,'gray');
ax4 = subplot(224); imagesc(fB); 
axis off; title('B'); colormap(ax4,'gray');
impixelinfo 

% Display the histograms
figure(4);
ax1 = subplot(221); imhist(f); 
axis off; title('gray'); colormap(ax1,'gray');
ax2 = subplot(222); imhist(fR); 
axis off; title('R'); colormap(ax2,'gray');
ax3 = subplot(223); imhist(fG); 
axis off; title('G'); colormap(ax3,'gray');
ax4 = subplot(224); imhist(fB); 
axis off; title('B'); colormap(ax4,'gray');
impixelinfo 


end

