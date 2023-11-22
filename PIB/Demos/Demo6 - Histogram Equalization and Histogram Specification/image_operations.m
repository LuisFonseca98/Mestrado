%
% ISEL - Instituto Superior de Engenharia de Lisboa. 
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Inform�tica e Multim�dia.
%   
% PIB - Processamento de Imagem e Biometria.
%
% image_operations.m
% Fun��o que ilustra a aplica��o de opera��es sobre imagens com n�veis de cinzento.
% Processamento espacial.
%
function image_operations()

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc 
 
% Ler a imagem a partir do ficheiro.
I = imread('camera.gif');
I2 = imread('lena.gif');
%I = imread('lena.gif');
%I = imread('f-18.bmp');
%I = imread('flowers.bmp');

% Obter as dimens�es (resolu��o da imagem).
[M, N] = size(I);

% Imprimir mensagem com as dimens�es e resolu��o da imagem.
fprintf('Imagem com resolu��o %d x %d = %d pixels\n', M, N, M*N);

% Lan�ar nova janela de figura e mostrar a imagem em n�veis de cinzento
% e o respetivo histograma.
% Vers�o negativa da imagem.
Ineg = 255 - I;
figure(1); set(gcf,'Name', 'Imagem negativa');
subplot(221); imshow(I); colorbar; title(' Imagem ' );
subplot(222); imshow(Ineg); colorbar; title(' Imagem negativa' );
subplot(223); imhist(I); title(' Histograma da Imagem ' );
subplot(224); imhist(Ineg); title(' Histograma da Imagem Negativa' );
impixelinfo;

% Soma de uma constante.
Iop= I + 80;
figure(2); set(gcf,'Name', 'Soma de constante: I + 80');
subplot(221); imshow(I); colorbar; title(' Imagem ' );
subplot(222); imshow(Iop); colorbar; title(' Imagem 1' );
subplot(223); imhist(I); title(' Histograma da Imagem ' );
subplot(224); imhist(Iop); title(' Histograma da Imagem 1' );
impixelinfo;

% Subtra��o de uma constante.
Iop= I - 80;
figure(3); set(gcf,'Name', 'Subtra��o de constante: I - 80');
subplot(221); imshow(I); colorbar; title(' Imagem ' );
subplot(222); imshow(Iop); colorbar; title(' Imagem 2' );
subplot(223); imhist(I); title(' Histograma da Imagem ' );
subplot(224); imhist(Iop); title(' Histograma da Imagem 2' );
impixelinfo;

% Amplifica��o.
Iop= 1.5*I;
figure(4); set(gcf,'Name', 'Amplifica��o: 1.5*I');
subplot(221); imshow(I); colorbar; title(' Imagem ' );
subplot(222); imshow(Iop); colorbar; title(' Imagem 3' );
subplot(223); imhist(I); title(' Histograma da Imagem ' );
subplot(224); imhist(Iop); title(' Histograma da Imagem 3' );
impixelinfo;
 
% AND 
Iop= bitand(I, I2);
figure(5); set(gcf,'Name', 'AND l�gico: I1 AND I2');
subplot(131); imshow(I); colorbar; title(' I1' );
subplot(132); imshow(I2); colorbar; title(' I2' );
subplot(133); imshow(Iop); colorbar; title(' I1 AND I2' );
impixelinfo;


% Equaliza��o de histograma.
Iop= histeq(I);
figure(6); set(gcf,'Name', 'Equaliza��o de histograma'); 
subplot(221); imshow(I); colorbar; title(' Imagem ' );
subplot(222); imshow(Iop); colorbar; title(' Imagem 4' );
subplot(223); imhist(I); title(' Histograma da Imagem ' );
subplot(224); imhist(Iop); title(' Histograma da Imagem 4' );
impixelinfo;

% Ajustar o contraste.
Iadj = imadjust(I);
figure(7); set(gcf,'Name', 'Ajuste de contraste');
subplot(221); imshow(I); colorbar; title(' Imagem ' );
subplot(222); imshow(Iadj); colorbar; title(' Imagem 5' );
subplot(223); imhist(I); title(' Histograma da Imagem ' );
subplot(224); imhist(Iadj); title(' Histograma da Imagem 5' );
impixelinfo;

end

