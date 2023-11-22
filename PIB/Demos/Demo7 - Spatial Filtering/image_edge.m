%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Inform�tica e Multim�dia.
% 
% PIB - Processamento de Imagem e Biometria.
%
% image_edge.m
% Fun��o que ilustra a aplica��o de opera��es sobre imagens com n�veis de cinzento.
% An�lise de exemplos de filtragem espacial passa-alto para dete��o de
% contornos.

function image_edge() 

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc
 
% Ler a imagem a partir do ficheiro.
%I = imread('lena.gif');
%I = imread('bird.gif');
I = imread('lena.gif');
%I = imread('f-18.bmp');
%I = imread('flowers.bmp');

% Dete��o de contornos por diferentes m�todos.
I1 = edge(I,'canny');    
I2 = edge(I,'roberts');    
I3 = edge(I,'log');    
I4 = edge(I,'sobel');    
I5 = edge(I,'prewitt');    

% Compara��o dos resultados com a imagem original.
figure(1); set(gcf,'Name', 'Edge detection ');
subplot(231); imagesc(I); title(' Imagem ' ); colormap('gray');
subplot(232); imagesc(I1); axis tight; title(' Canny' );
subplot(233); imagesc(I2); axis tight; title(' Roberts' );
subplot(234); imagesc(I3); axis tight; title(' Laplacian of Gaussian' );
subplot(235); imagesc(I4); axis tight; title(' Sobel' );
subplot(236); imagesc(I5); axis tight; title(' Prewitt' );
impixelinfo;
    
end



