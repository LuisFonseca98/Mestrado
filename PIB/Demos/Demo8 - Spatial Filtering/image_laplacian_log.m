% 
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%  
% PIB - Processamento de Imagem e Biometria.
%
% image_laplacian_log.m
% Função que ilustra a aplicação de operações sobre imagens com níveis de cinzento.
% Análise de exemplos de filtragem espacial passa-alto para deteção de
% contornos. Comparação dos resultados de Laplaciano com LoG - Laplacian of
% Gaussian.
  
function image_laplacian_log()

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc

% Ler a imagem a partir do ficheiro.
I = imread('circuit.tif');
%I = imread('spine.tif');
%I = imread('rice.png');

%I = imnoise(I,'gaussian',0,0.001);
M = size(I,1);
N = size(I,2);
noise1 = zeros( M, N );
for m=1 : M
    for n=1 : N
        noise1(m,n) =100*cos( (pi/30)*m ) * cos( (pi/30)*n );   
    end
end
I = 0.85*double(I) + 0.15*noise1;
I = uint8(I);


% Laplaciano 1 - definição
L1 = [  0 1 0; 
           1 -4 1; 
           0 1 0];

% Laplaciano 2 - definição
L2 = [  1 1 1; 
           1 -8 1; 
           1 1 1];

% h = fspecial('log', hsize, sigma) 
%        returns a rotationally symmetric Laplacian of Gaussian filter 
%        of size hsize with standard deviation sigma (positive). 
%        hsize can be a vector specifying the number of rows and columns in h, 
%        or it can be a scalar, in which case h is a square matrix. 
%        The default value for hsize is [5 5] and 0.5 for sigma.
LoG = fspecial('log', [5, 5], 0.15 );

% Imagem melhorada.
I1      = filter2(L1, I);
I2      = filter2(L2, I);
I_LoG   = filter2(LoG,I);

% Comparação dos resultados com a imagem original.
figure(1); set(gcf,'Name', 'Laplaciano - imagens');
subplot(221); imshow(I); title(' Original' ); colormap('gray'); 
subplot(222); imshow(uint8(I1)); axis tight; title(' Laplacian 1' ); 
subplot(223); imshow(uint8(I2)); axis tight; title(' Laplacian 2' ); 
subplot(224); imshow(uint8(I_LoG)); axis tight; title(' LoG' ); 
impixelinfo;
% 
% figure(2);
% imshow(uint8(I1)); axis tight; title(' Laplaciano 1' ); colorbar;
% 
% figure(3);
% imshow(uint8(I2)); axis tight; title(' Laplaciano 2' ); colorbar;

% Análise do efeito da variação de sigma (desvio padrão) no LoG
I = imread('spine.tif');
figure(2);
for sigma= 0.01 : 0.01 : 0.55
    LoG = fspecial('log', [5, 5], sigma );
    I_LoG = filter2(LoG,I);
    
    figure(2); set(gcf,'Name', 'Log - images');
    subplot(121); imshow(I); title(' Original' ); colormap('gray'); colorbar;
    subplot(122); imshow(uint8(I_LoG)); axis tight; title( sprintf(' LoG, sigma=%.2f', sigma) ); colorbar;
    pause(1);
end
 
end



