%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%  
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria.
% 
% image_contrast.m
% Função que ilustra a aplicação de operações sobre imagens com níveis de cinzento.
% Processamento espacial.
% Análise de imagens com diferentes valores de contraste.
%
function image_contrast()

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc 

% Ler a imagem a partir do ficheiro.
I_low  = imread('low.tif');
I_med  = imread('med.tif');
I_high = imread('high.tif');

figure(1); set(gcf,'Name', 'Contrast');
subplot(231); imshow(I_low);  colorbar; colormap('gray'); title(' Low '    );
subplot(232); imshow(I_med);  colorbar; colormap('gray'); title(' Medium ' );
subplot(233); imshow(I_high); colorbar; colormap('gray'); title(' High '   );

subplot(234); imhist(I_low);  colorbar; colormap('gray'); title(' Low '    );
subplot(235); imhist(I_med);  colorbar; colormap('gray'); title(' Medium ' );
subplot(236); imhist(I_high); colorbar; colormap('gray'); title(' High '   );
impixelinfo;

% Ajustar o contraste.
%Iadj = imadjust(I_low);
Iadj = imadjust(I_low, [0.3; 0.6], [0.1; 0.9] );
figure(2); set(gcf,'Name', 'Ajuste de contraste');
subplot(221); imshow(I_low); colorbar; colormap('gray'); title(' Low ' );
subplot(222); imshow(Iadj);  colorbar; colormap('gray'); title(' Low with Adjust' );
subplot(223); imhist(I_low); colorbar; colormap('gray'); title(' Low ' );
subplot(224); imhist(Iadj);  colorbar; colormap('gray'); title(' Low with Adjust' );
impixelinfo;

mx1 = double( max(max(I_low)) ) + 1;
mi1  = double( min(min(I_low)) ) + 1;
EME1 = 20 * log( mx1 / mi1 )

mx2 = double( max(max(I_med)) ) + 1;
mi2 = double( min(min(I_med)) ) + 1;
EME2 = 20 * log( mx2 / mi2 )

mx3 = double( max(max(I_high)) ) + 1;
mi3  = double( min(min(I_high)) ) + 1;
EME3 = 20 * log( mx3 / mi3 )

mx4 = double( max(max(Iadj)) ) + 1;
mi4  = double( min(min(Iadj)) ) + 1;
EME4 = 20 * log( mx4 / mi4 )

m1 = mean(mean(I_low));
m2 = mean(mean(I_med));
m3 = mean(mean(I_high));
m4 = mean(mean(Iadj));


figure(3);
subplot(1,2,1);
bar( [m1, m2, m3, m4]' );
grid on;
title( sprintf('Brilho:  Low=%.2f, Medium=%.2f, High=%.2f, Low-Adjusted=%.2f ',...
    m1, m2, m3, m4) );

subplot(1,2,2);
bar( [EME1, EME2, EME3, EME4]' );
grid on;
title( sprintf('Contraste:  Low=%.2f, Medium=%.2f, High=%.2f, Low-Adjusted=%.2f ',...
    EME1, EME2, EME3, EME4) );
end

