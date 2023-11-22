%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Inform�tica e Multim�dia.
% 
% PIB - Processamento de Imagem e Biometria.
%  
% image_negative_binarization.m
% Fun��o que ilustra a aplica��o de opera��es sobre imagens com n�veis de cinzento.
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

% Calcular o limiar �timo para transformar a imagem 
% na sua vers�o bin�ria (m�todo de Otsu).
level = graythresh(I);

% Converter para imagem bin�ria.
IBW = imbinarize(I, level);

% Lan�ar nova janela de figura e mostrar as imagens em n�veis de cinzento
% e bin�ria.
figure(1); set(gcf, 'Name', ['Original e bin�ria - ' num2str(255*level)] );
subplot(121); imshow(I);   colorbar; title(' Imagem ' );
subplot(122); imshow(IBW); colorbar; title(' Imagem Bin�ria' );
impixelinfo;
imwrite(IBW,'eight_bw.gif','gif');

% Vers�o negativa e respetiva binariza��o.
I = 255 - I;
level = graythresh(I);
IBW = imbinarize(I, level);

% Lan�ar nova janela de figura e mostrar as imagens em n�veis de cinzento
% e bin�ria.
figure(2); set(gcf,'Name', 'Negativa e bin�ria');
subplot(121); imshow(I);   colorbar; title(' Imagem ' );
subplot(122); imshow(IBW); colorbar; title(' Imagem Bin�ria' );
impixelinfo;
imwrite(IBW,'eight_bw2.gif','gif');
end

