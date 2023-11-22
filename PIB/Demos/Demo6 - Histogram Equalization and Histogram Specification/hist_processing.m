%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%  
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
%
% hist_processing.m
% Função que compara os efeitos de equalização e especificação de 
% histograma. 
 
function hist_processing()
 
% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc

% Ler a imagem a partir do ficheiro.
I     = imread('tire.tif');
%I     = imread('moon_phobos.tif');
Ieq = histeq(I); % HE

Iref    = imread('lena.gif');
% Iref    = imread('low.tif');
%Iref    = imadjust(Iref);
Ispec = imhistmatch(I, Iref ); %HS

figure(1); set(gcf, 'Name', 'Processamento de Histograma: Equalização vs Especificação');
subplot(231); imshow(I); title(' Imagem ' );
subplot(232); imshow(Ieq); title(' Imagem - Histograma Equalizado' );
subplot(233); imshow(Ispec); title(' Imagem - Histograma Especificado' );

subplot(234); imhist(I); title(' Imagem ' );
subplot(235); imhist(Ieq); title(' Imagem - Histograma Equalizado' );
subplot(236); imhist(Ispec); title(' Imagem - Histograma Especificado' );
impixelinfo;
 
figure(2); set(gcf, 'Name', 'Processamento de Histograma: Equalização vs Especificação');
subplot(121); imhist(Iref); title(' Histograma Especificado/Referência ' );
subplot(122); imhist(Ispec); title(' Histograma Obtido' );
impixelinfo;

Iadj = imadjust(I);
figure(3); set(gcf, 'Name', 'Processamento de Histograma: Equalização vs Especificação');
subplot(221); imshow(I); title(' Imagem ' );
subplot(222); imshow(Iadj); title(' Imagem - Ajuste' );

subplot(223); imhist(I); title(' Imagem - Histograma' );
subplot(224); imhist(Iadj); title(' Imagem - Ajuste - Histograma ' );
impixelinfo;

end

