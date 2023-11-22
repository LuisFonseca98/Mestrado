% 
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria.
%
% image_laplacian_enhancement.m
% Função que ilustra a aplicação de melhoria de imagem com Laplaciano.
  
function image_laplacian_enhancement()

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc

% Ler a imagem a partir do ficheiro.
I = imread('circuit.tif');
%I = imread('spine.tif');
%I = imread('squares.gif');

% Laplaciano 1 - para melhoria de imagem
L1 = [  0 -1 0; 
        -1 5 -1; 
        0 -1 0];

% Laplaciano 2 - para melhoria de imagem
L2 = [  -1 -1 -1; 
        -1 9 -1; 
        -1 -1 -1];

% Imagem melhorada.
I_en1 = filter2(L1, I);
I_en2 = filter2(L2, I);

% Comparação dos resultados com a imagem original.
% Imagens.
figure(1); set(gcf,'Name', 'Laplacian - image');
subplot(231);  imshow(I); title(' Original' ); colormap('gray'); 
subplot(232);  imshow(uint8(I_en1)); axis tight; title(' Laplacian 1' ); 
subplot(233);  imshow(uint8(I_en2)); axis tight; title(' Laplacian 2' ); 
subplot(234);  imhist(I); title(' Original' ); 
subplot(235);  imhist(uint8(I_en1));  title(' Laplacian 1' ); 
subplot(236);  imhist(uint8(I_en2)); title(' Laplacian 2' ); 
end



