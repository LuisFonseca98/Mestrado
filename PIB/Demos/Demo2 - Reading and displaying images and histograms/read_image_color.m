%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informática e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
%  
% read_image_color.m
% Função que ilustra a manipulação de imagens a cores.
 
function read_image_color() 
 
% Fechar todas as janelas de figuras.
close all;

% Ler a imagem a cores, a partir do ficheiro.
%I = imread('football.bmp');
I = imread('barries.tif');
%I = imread('peppers.tif');
%I = imread('flowers.bmp');

% Obter as componentes R, G, e B.
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);

% Lançar nova janela de figura e mostrar a imagem original a cores
% e as componentes R, G e B como imagens em níveis de cinzento.
figure(1);
subplot(221); imshow(I);  colorbar; title(' Color [R, G, B]' );
subplot(222); imshow(R); colorbar; title(' Red component' );
subplot(223); imshow(G); colorbar; title(' Green component' );
subplot(224); imshow(B); colorbar; title(' Blue component' );

% Ativar o cursor que mostra as coordenadas (x,y) e 
% o valor do pixel nessas coordenadas.
impixelinfo;
 
% Escrever a imagem a cores como ficheiro JPEG
imwrite( I, 'football2.jpg' );
 
% Converter a imagem a cores para uma versão com níveis de cinzento.
Ig = rgb2gray( I ); 
% rgb2gray converts RGB values to grayscale values by forming a 
% weighted sum of the R, G, and B components:
% Y = 0.2989 * R + 0.5870 * G + 0.1140 * B 

% Lançar nova janela de figura para mostrar a imagem a cores
% e a imagem com níveis de cinzento.
figure(2);
subplot(211); imshow(I); colorbar; title(' Color [R, G, B]' );
subplot(212); imshow(Ig); colorbar; title(' Gray Levels' );
impixelinfo;
end  

