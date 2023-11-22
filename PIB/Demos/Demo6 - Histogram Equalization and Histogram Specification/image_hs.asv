%
% ISEL - Instituto Superior de Engenharia de Lisboa.
% 
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
% 
% image_hs.m
% Função que ilustra a aplicação da especificação de histograma.
%
function image_hs()

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc 

% Ler a imagem a partir do ficheiro.
I    = imread('low.tif');
Iref = imread('lena.gif');

% Especificação de histograma
J = imhistmatch(I, Iref);
  
% Mostrar as imagens 
subplot(231); imshow(I); title('Original Image');
subplot(232); imshow(Iref); title('Reference Image');
subplot(233); imshow(J); title('Image after Histogram Specification');

% Mostrar os histogramas
subplot(234); imhist(I); title('Original Image');
subplot(235); imhist(Iref); title('Reference Image');
subplot(236); imhist(J); title('Image after Histogram Specification');

end

