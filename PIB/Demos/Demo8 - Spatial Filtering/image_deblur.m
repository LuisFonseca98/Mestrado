% 
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
%  
% image_deblur.m
% Função que ilustra a aplicação de melhoria de imagem com Laplaciano.
  
function image_deblur()

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc

% Ler a imagem a partir do ficheiro.
%I = imread('circuit.tif');
I = imread('bird_blurred.png');
%I = imread('squares.gif');

% Laplaciano 1 - para melhoria de imagem
L1 = [  0 -1 0; 
        -1 5 -1; 
        0 -1 0];

% Laplaciano 2 - para melhoria de imagem
L2 = [  -1 -1 -1; 
        -1 9 -1; 
        -1 -1 -1];
    
Id1 = filter2(L1, I);
Id2 = filter2(L2, I);

% Comparação dos resultados com a imagem original.
% Imagens.
figure(1); set(gcf,'Name', 'Sharpening with the Laplacian');
subplot(131);  imshow(I); title(' Original (blurred)' ); colormap('gray'); 
subplot(132);  imshow(uint8(Id1)); axis tight; title(' Sharpening 1 ' ); 
subplot(133);  imshow(uint8(Id2)); axis tight; title(' Sharpening 2 ' ); 

print '-dpng' 'deblur.png'
end



