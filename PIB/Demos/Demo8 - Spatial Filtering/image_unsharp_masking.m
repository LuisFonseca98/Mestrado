% 
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
%
% image_unsharp_masking.m
% The unsharp_masking technique.
  
function image_unsharp_masking()
 
% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc

% Ler a imagem a partir do ficheiro.
I = imread('bird.gif');
%I = imread('spine.tif');
%I = imread('squares.gif');

G = fspecial('gaussian', [7 7],  2.5); 
 %h = fspecial('gaussian', hsize, sigma) 
 % returns a rotationally symmetric Gaussian lowpass filter 
 % of size hsize with standard deviation sigma (positive). 
 % hsize can be a vector specifying the number of rows and columns in h, 
 % or it can be a scalar, in which case h is a square matrix. 
 %The default value for hsize is [3 3]; the default value for sigma is 0.5.

% Step 1 - Blurred image.
Ib = filter2(G, I);

% Step 2 - Get the mask.
IMask = double(I) - double(Ib);
mi = min(min(IMask))
mx = max(max(IMask))

% Step 3 - Sum the mask to the original window.
I_un = double(I) + IMask;

% Comparison.
figure(1); set(gcf,'Name', 'Image Unsharp Masking');
subplot(221); imshow(I); title(' Original' ); colormap('gray'); colorbar;
subplot(222); imshow(uint8(Ib)); axis tight; title(' Blurred' ); colorbar;
subplot(223); imagesc(IMask); axis tight; title(' Mask' ); colorbar;
subplot(224); imshow(uint8(I_un)); axis tight; title(' Enhanced ' ); colorbar;
% subplot(223); imshow(I); title(' Original' ); colormap('gray'); colorbar;
% subplot(224); imshow(uint8(I_en2)); axis tight; title(' Melhorada 2' ); colorbar;
impixelinfo;

% Comparação dos resultados com a imagem original.
% Histogramas.
figure(2); set(gcf,'Name', 'Image Unsharp Masking');
subplot(221); imhist(I); title(' Original' ); 
subplot(222); imhist(uint8(Ib)); axis tight; title(' Blurred' ); 
subplot(223); imhist(IMask); axis off; title(' Máscara' ); 
subplot(224); imhist(uint8(I_un)); axis tight; title(' Melhorada' ); 

% Análise do efeito da variação de k 
for k= 0.1 : 0.1 : 4
    figure(3); set(gcf,'Name', 'Image HighBoost Filtering');
    % Passo 3 - Somar a máscara à imagem original
    I_un = double(I) + k*IMask;
    subplot(121); imshow(I); title(' Original' ); colormap('gray'); colorbar;
    subplot(122); imshow(uint8(I_un)); axis tight; title( sprintf(' k=%.2f', k ) ); colorbar;
    pause(2); impixelinfo;
end

end



