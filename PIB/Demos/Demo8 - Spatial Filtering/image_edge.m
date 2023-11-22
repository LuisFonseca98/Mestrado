% 
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Inform�tica e Multim�dia.
%
% PIB - Processamento de Imagem e Biometria.
% 
% image_edge.m
% Fun��o que ilustra a aplica��o de opera��es de dete��o de
% contornos, atrav�s de diferentes algoritmos.
  
function image_edge()

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc

% Ler a imagem a partir do ficheiro.
%I = imread('lena.gif');
I = imread('bird.gif');
% I = imread('squares.gif');
%I = imread('JCanny.png');
%I = rgb2gray(I);
%I = imread('lena.gif');
%I = imread('f-18.bmp');
%I = imread('flowers.bmp');

% Dete��o de contornos por diferentes m�todos.
I1 = edge(I,'canny');    
I2 = edge(I,'roberts');    
I3 = edge(I,'log');    
I4 = edge(I,'sobel');    
I5 = edge(I,'prewitt');    

% Compara��o dos resultados com a imagem original.
figure(1); set(gcf,'Name', 'Edge Detection');
subplot(231); imagesc(I); title(' Image ' ); colormap('gray'); axis off;
subplot(232); imagesc(I1); axis tight; title(' Canny' ); axis off;
subplot(233); imagesc(I2); axis tight; title(' Roberts' ); axis off;
subplot(234); imagesc(I3); axis tight; title(' Laplacian of Gaussian' ); axis off;
subplot(235); imagesc(I4); axis tight; title(' Sobel' ); axis off;
subplot(236); imagesc(I5); axis tight; title(' Prewitt' ); axis off;
print '-dpng' 'edge1.png'

% Compara��o dos resultados com a imagem original.
figure(2); set(gcf,'Name', 'Edge Detection');
subplot(121); imagesc(I); title(' Image ' ); colormap('gray'); axis off;
subplot(122); imagesc(I1); axis tight; title(' Canny' ); axis off;
print '-dpng' 'edge2.png'

end



