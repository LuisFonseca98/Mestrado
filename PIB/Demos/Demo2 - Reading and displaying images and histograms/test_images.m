%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Inform�tica e Multim�dia.
% 
% PIB - Processamento de Imagem e Biometria
%
% test_images.m  
%
 
function test_images()

% Fechar todas as janelas com figuras. 
close all

% Ler a imagem a cores (RGB).
I  = imread('football2.jpg');

% Converter a imagem a cores na sua vers�o em n�veis de cinzento.
Ig = rgb2gray(I);

% Converter a imagem em n�veis de cinzento, para a sua vers�o bin�ria.
% Calcular o limiar (threshold, th) �timo para a convers�o.
th = graythresh(Ig);
Ib = imbinarize(Ig, th);

% Mostrar as tr�s vers�es da imagem.
figure;
subplot(131); imshow(I);  title(' Colorida - RGB '); %impixelinfo;
subplot(132); imshow(Ig); title(' Niveis de cinzento '); %impixelinfo;
subplot(133); imshow(Ib); title(' Bin�ria '); %impixelinfo;

% Mostrar as imagens em n�veis de cinzento e bin�rias e 
% respetivos histogramas.
figure;
subplot(221); imshow(Ig); title(' Niveis de cinzento ');
subplot(222); imhist(Ig); title(' Histograma - Niveis de cinzento ');
subplot(223); imshow(Ib); title(' Bin�ria ');
subplot(224); imhist(Ib); title(' Histograma - Bin�ria ');

% Mostrar os histogramas das componentes R, G, B da imagem a cores.
figure;
subplot(131); imhist(I(:,:,1));  title(' Histograma - R ');
subplot(132); imhist(I(:,:,2));  title(' Histograma - G ');
subplot(133); imhist(I(:,:,3));  title(' Histograma - B ');
return

