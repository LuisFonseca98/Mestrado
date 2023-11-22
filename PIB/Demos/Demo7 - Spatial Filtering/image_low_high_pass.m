%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria.
%
% image_low_high_pass.m
% Função que ilustra a aplicação de operações sobre imagens com níveis de cinzento.
% Análise de exemplos de filtragem espacial passa-baixo e passa-alto.
  
function image_low_high_pass()
 
% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc

% Ler a imagem a partir do ficheiro.
I = imread('lena.gif');
%I = imread('bird.gif');
%I = imread('lena.gif');
%I = imread('f-18.bmp');
%I = imread('flowers.bmp');

% Obter as dimensões (resolução da imagem).
[M, N] = size(I);

% Imprimir mensagem com as dimensões e resolução da imagem.
fprintf('Image with %d x %d = %d pixels\n', M, N, M*N);

% Filtro 1
% Filtro passa-baixo de 3x3
k = (1/9) * ones(3,3);
I1 = filter2(k, I);

% Filtro 2
% Filtro passa-alto de 3x3 (complementar do passa-baixo).
k2 = zeros(3,3);
k2(2,2) = 1;
k2 = k2 - k;
I2  = filter2(k2,I);

% 
% Mostrar as imagens resultantes. 
%
figure(1); set(gcf,'Name', 'Filtering');
subplot(221); imagesc(I); title(' Image ' ); colormap('gray');
subplot(222); imagesc(I1); axis tight; title(' Low-pass  (smoothing)' );
subplot(223); imagesc(I2); axis tight; title(' High-pass (edges)' );
impixelinfo;

% 
% Mostrar as imagens resultantes. 
%
figure(2); set(gcf,'Name', 'Filtering');
subplot(121); imagesc(I); title(' Image ' ); colormap('gray');
subplot(122); imagesc(0.8*double(I)+0.2*double(I2)); 
axis tight; title(' Image + High-pass = Sharpening' );
impixelinfo;
 
% 
% Mostrar as imagens resultantes. 
%
figure(3); set(gcf,'Name', 'Filtering');
subplot(121); imagesc(I); title(' Image ' ); 
colormap('gray');
subplot(122); imagesc(double(I) + 0.5*(double(I)-double(I1))); 
axis tight; title(' Imagem - Sharpened 1' );
impixelinfo;

% 
% Mostrar as imagens resultantes. 
%
figure(4); set(gcf,'Name', 'Filtering');
subplot(121); imagesc(I); title(' Image ' ); 
colormap('gray');
subplot(122); imagesc(imsharpen(I)); 
axis tight; title(' Imagem - Sharpened 2' );
impixelinfo;

end

 