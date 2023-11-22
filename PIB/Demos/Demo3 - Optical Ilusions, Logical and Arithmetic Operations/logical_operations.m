%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%  
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria.
%
% logical_operations.m
% Função que ilustra o efeito de operações lógicas sobre imagens.
 
function logical_operations() 

% Fechar todas as janelas de figuras.
close all;
 
% Limpar a consola.
clc
 
% Ler a imagem a partir do ficheiro.
I1 = imread('lena.gif');
%I2 = imread('squares.gif');
I2 = imread('camera.gif');

I3 = bitand(I1, I2);
close all
figure(1);
subplot(1,3,1); imshow(I1); title('I1');
subplot(1,3,2); imshow(I2); title('I2');
subplot(1,3,3); imshow(I3); title('I1 AND I2');

I4 = bitor(I1, I2);
figure(2);
subplot(1,3,1); imshow(I1); title('I1');
subplot(1,3,2); imshow(I2); title('I2');
subplot(1,3,3); imshow(I4); title('I1 OR I2');

I5 = bitxor(I1, I2);
figure(3);
subplot(1,3,1); imshow(I1); title('I1');
subplot(1,3,2); imshow(I2); title('I2');
subplot(1,3,3); imshow(I5); title('I1 XOR I2');

I6 = bitxor(I1, 255);
I7 = bitxor(I2, 255);
figure(4);
subplot(2,2,1); imshow(I1); title('I1');
subplot(2,2,2); imshow(I2); title('I2');
subplot(2,2,3); imshow(I6); title('NOT I1');
subplot(2,2,4); imshow(I7); title('NOT I2');
end

