%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informática e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
% 
% read_image_color_v2.m
% Função que ilustra a manipulação de 
% imagens a cores.
  
function read_image_color_v2() 

% Fechar todas as janelas de figuras.
close all;

% Ler a imagem a partir do ficheiro.
I = imread('football.bmp');
%I = imread('barries.tif');
%I = imread('peppers.tif');

% Obter as componentes R, G, e B.
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);

% Lançar nova janela de figura e mostrar a imagem original a cores
% e as componentes R, G e B como imagens em níveis de cinzento.
figure(1);
subplot(221); imshow(I);  colorbar; title(' Color [R, G, B]' );
subplot(222); imshow(R); colorbar; title(' Red component' );
subplot(223); imshow(G); colorbar; title(' Green component' );
subplot(224); imshow(B); colorbar; title(' Blue component' );
impixelinfo;

% Energia por banda R, G e B
ER = sum(sum(double(R).^2))
EG = sum(sum(double(G).^2))
EB = sum(sum(double(B).^2))
% 
% % Energia diretamente em RGB
% E = sum(sum(double(I).^2))

% Trocar as componentes R e G
I2 = I;
S = I2(:,:,1);
I2(:,:,1) = I2(:,:,2);
I2(:,:,2) = S;

% Lançar nova janela de figura e mostrar a imagem resultante
figure(2);
subplot(211); imshow(I); colorbar; title(' I1' );
subplot(212); imshow(I2); colorbar; title(' I2' );
impixelinfo;
 
% Trocar as componentes R e B
I3 = I; 
S = I3(:,:,1);
I3(:,:,1) = I2(:,:,3);
I3(:,:,3) = S;
 
% Lançar nova janela de figura e mostrar a imagem resultante
figure(3);
subplot(211); imshow(I); colorbar; title(' I1' );
subplot(212); imshow(I3); colorbar; title(' I3' );
impixelinfo;

end  
