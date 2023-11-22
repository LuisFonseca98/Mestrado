%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria.
%  
% image_negative_binarization.m
% Função que ilustra a aplicação de operações sobre imagens com níveis de cinzento.
% Processamento espacial.
%
function image_negative_binarization() 

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc

% Ler a imagem a partir do ficheiro.
I = imread('eight.tif');
%I = imread('finger.tif');

% Calcular o limiar ótimo para transformar a imagem 
% na sua versão binária (método de Otsu).
level = graythresh(I);

% Converter para imagem binária.
IBW = imbinarize(I, level);

% Lançar nova janela de figura e mostrar as imagens em níveis de cinzento
% e binária.
figure(1); set(gcf, 'Name', ['Original e binária - ' num2str(255*level)] );
subplot(121); imshow(I);   colorbar; title(' Imagem ' );
subplot(122); imshow(IBW); colorbar; title(' Imagem Binária' );
impixelinfo;
imwrite(IBW,'eight_bw.gif','gif');

% Versão negativa e respetiva binarização.
I = 255 - I;
level = graythresh(I);
IBW = imbinarize(I, level);

% Lançar nova janela de figura e mostrar as imagens em níveis de cinzento
% e binária.
figure(2); set(gcf,'Name', 'Negativa e binária');
subplot(121); imshow(I);   colorbar; title(' Imagem ' );
subplot(122); imshow(IBW); colorbar; title(' Imagem Binária' );
impixelinfo;
imwrite(IBW,'eight_bw2.gif','gif');
end

