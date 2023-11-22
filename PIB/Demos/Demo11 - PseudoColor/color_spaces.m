%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%   
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%   
% PIB - Processamento de Imagem e Biometria.
%
% color_spaces.m
% Análise de espaços de cor.
%
%

function color_spaces()

% Fechar todas as janelas com figuras.
close all;

% Ler a imagem para memoria
%I = imread('cameraman.tif');
%I = imread('bo256256.bmp');
 
% Ler a imagem RGB para uma matriz (M x N x 3).
%I_RGB = imread('football.bmp');
%I_RGB = imread('monarch.tif');
%I_RGB = imread('tulips.tif');
%I_RGB = imread('clegg.tif');
I_RGB = imread('peppers.png');

% Converter para imagem com níveis de cinzento (M x N x 1).
Ig = rgb2gray(I_RGB);
% rgb2gray converts RGB values to grayscale values by forming a weighted sum of the R, G, and B components:
%Y = 0.2989 * R + 0.5870 * G + 0.1140 * B 

% Mostrar imagem RGB e com níveis de cinzento.
figure(1);
subplot(121); imshow(I_RGB); title('RGB');
subplot(122); imshow(Ig); title('Gray Scale');

% Mostrar as componentes R, G, B como imagens individuais em níveis de cinzento.
figure(2);
subplot(221); imshow(I_RGB);
colormap('gray');
subplot(222); imagesc(I_RGB(:,:,1)); title('R');
subplot(223); imagesc(I_RGB(:,:,2)); title('G');
subplot(224); imagesc(I_RGB(:,:,3)); title('B');

% Análise da energia das várias componentes R, G e B.
ER = sum (sum(I_RGB(:,:,1).^2))
EG = sum (sum(I_RGB(:,:,2).^2))
EB = sum (sum(I_RGB(:,:,3).^2))


% % Converter a imagem RGB na respetiva imagem YIQ.
% % Y = Luminância
% % I, Q = Crominâncias.
% % http://en.wikipedia.org/wiki/YIQ
% % http://en.wikipedia.org/wiki/NTSC
% I_YIQ = rgb2ntsc(I_RGB);
% figure(3);
% subplot(221); imshow(I_YIQ);
% colormap('gray');
% subplot(222); imagesc(I_YIQ(:,:,1)); title('Y');
% subplot(223); imagesc(I_YIQ(:,:,2)); title('I');
% subplot(224); imagesc(I_YIQ(:,:,3)); title('Q');

% Converter a imagem RGB na respetiva imagem YCbCr.
% Y = Luminância
% Cb, Cr = Crominâncias.
% http://en.wikipedia.org/wiki/YCbCr
I_YCbCr = rgb2ycbcr(I_RGB);
figure(3);
subplot(221); imshow(I_RGB);
colormap('gray');
subplot(222); imagesc(I_YCbCr(:,:,1)); title('Y');
subplot(223); imagesc(I_YCbCr(:,:,2)); title('Cb');
subplot(224); imagesc(I_YCbCr(:,:,3)); title('Cr');
% 
% 
% Converter a imagem RGB na respetiva imagem HSV.
% H = Hue.
% S = Saturation.
% V = Value.
% http://en.wikipedia.org/wiki/HSL_and_HSV
I_HSV = rgb2hsv(I_RGB);
figure(5);
subplot(221); imshow(hsv2rgb(I_HSV));
colormap('gray');
subplot(222); imagesc(I_HSV(:,:,1)); title('H');
subplot(223); imagesc(I_HSV(:,:,2)); title('S');
subplot(224); imagesc(I_HSV(:,:,3)); title('V');

return
 
