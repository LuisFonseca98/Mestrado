%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
%  
% test_intensity_slicing.m
% Analysis of the intensity slicing technique.
 
function test_intensity_slicing()

% Clear console.
clc

% Close all windows
close all

% Input image
%f = imread('squares.gif');
%f = imread('bird.gif');
%f = imread('circuit.tif');
%f = imread('weld.tif');
%f = imread('xray.tif');
%f = imread('xray_neg.tif');
f = imread('phantom.tif');
f = imadjust(f); % Contrast enhancement
 
figure(1);
ax1 = subplot(221); imagesc(f); 
colormap(ax1,'gray'); axis off; title('gray');
ax2 = subplot(222); imagesc(f); 
colormap(ax2,'summer'); axis off; title('summer');
ax3 = subplot(223); imagesc(f); 
colormap(ax3,'jet'); axis off; title('jet');
ax4 = subplot(224); imagesc(f); 
colormap(ax4,'winter'); axis off; title('winter');
impixelinfo 

figure(2);
% Intensity slicing
my_map = colormap('jet'); 
f_RGB  = ind2rgb(f, my_map);ax1 = subplot(121); imagesc(f); 
axis off; title('gray'); colormap(ax1,'gray'); 
subplot(122); imagesc(f_RGB); 
axis off; title('RGB'); colorbar;
impixelinfo 

f_RGB = uint8(255*f_RGB);
figure(3);
ax1 = subplot(121); imagesc(f); 
axis off; title('gray'); colormap(ax1,'gray'); 
subplot(122); imshow(f_RGB); 
axis off; title('RGB'); colorbar;
impixelinfo 
end

