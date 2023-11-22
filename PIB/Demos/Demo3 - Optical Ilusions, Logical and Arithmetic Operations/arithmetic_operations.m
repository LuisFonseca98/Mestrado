%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%   
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%  
% PIB - Processamento de Imagem e Biometria.
%
% arithmetic_operations.m
% Função que ilustra o efeito de operações aritméticas sobre imagens.
 
function arithmetic_operations() 

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc

% Ler a imagem a partir do ficheiro.
I1 = imread('lena.gif');
%I2 = imread('squares.gif');
I2 = imread('camera.gif');
%I3 = I1 + I2;
I3 = uint8( 0.5*double(I1) + 0.5*double(I2) );

close all
figure(1);
subplot(2,3,1); imshow(I1); title('I1');
subplot(2,3,2); imshow(I2); title('I2');
subplot(2,3,3); imshow(I3); title('I1 + I2');
subplot(2,3,4); imhist(I1); title('I1');
subplot(2,3,5); imhist(I2); title('I2');
subplot(2,3,6); imhist(I3); title('I1 + I2');

I4 = I1 - I2;
figure(2);
subplot(2,3,1); imshow(I1); title('I1');
subplot(2,3,2); imshow(I2); title('I2');
subplot(2,3,3); imshow(I4); title('I1 - I2');
subplot(2,3,4); imhist(I1); title('I1');
subplot(2,3,5); imhist(I2); title('I2');
subplot(2,3,6); imhist(I4); title('I1- I2');

I5 = uint8(double(I1) .* double(I2));
figure(3);
subplot(2,3,1); imshow(I1); title('I1');
subplot(2,3,2); imshow(I2); title('I2');
subplot(2,3,3); imshow(I5); title('I1 * I2');
subplot(2,3,4); imhist(I1); title('I1');
subplot(2,3,5); imhist(I2); title('I2');
subplot(2,3,6); imhist(I5); title('I1 * I2');

I6 = uint8(double(I1)  + 30);
figure(4);
subplot(2,2,1); imshow(I1); title('I1');
subplot(2,2,2); imshow(I6); title('I1 + 50');
subplot(2,2,3); imhist(I1); title('I1');
subplot(2,2,4); imhist(I6); title('I1 + 50');

I7 = uint8(double(I1)  * 2);
figure(5);
subplot(2,2,1); imshow(I1); title('I1');
subplot(2,2,2); imshow(I7); title('2 * I1');
subplot(2,2,3); imhist(I1); title('I1');
subplot(2,2,4); imhist(I7); title('2* I1');
end

