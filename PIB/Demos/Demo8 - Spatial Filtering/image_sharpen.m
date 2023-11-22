% 
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
%
% image_sharpen.m
% Função que ilustra a aplicação da técnica de unsharp masking 
% sobre imagens com níveis de cinzento.
   
function image_sharpen()

% Fechar todas as janelas de figuras.
close all; 

% Limpar a consola.
clc
I    = imread('bird.gif');
L   = 5;
w  = ones(L,L) / L^2;
Ib  = filter2(w,I);
Ib  = uint8(Ib);
imwrite(Ib,'bird_blurred.png','png');

Iu1 = imsharpen(Ib);
Iu2 = imsharpen(Ib,'radius',2.5);

figure(1);
subplot(221); imshow(I); title('Original');
subplot(222); imshow(Ib); title('Iblurred');
subplot(223); imshow(Iu1); title('Isharpened');
subplot(224); imshow(Iu2); title('Isharpened 2');

end
