%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%   
% PIB - Processamento de Imagem e Biometria.
%
% image_negative.m
% Função que ilustra a aplicação de operações sobre imagens com níveis de cinzento.
% Processamento espacial.
% Análise de imagens e do seu negativo.
% 
function image_negative()

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc 
  
% Ler a imagem a partir do ficheiro.
I = imread('xray.tif');
In = 255 - I;
imwrite(In, 'xray_neg.tif');

figure(1); set(gcf,'Name', 'Imagem e seu negativo');
subplot(221); imshow(I);  colorbar; title(' Imagem ' ); colormap('gray');
subplot(222); imshow(In); colorbar; title(' Negativo ' ); colormap('gray');

subplot(223); imhist(I);  colorbar; title(' Imagem  ' ); colormap('gray');
subplot(224); imhist(In); colorbar; title(' Negativo ' ); colormap('gray');
impixelinfo;
end

