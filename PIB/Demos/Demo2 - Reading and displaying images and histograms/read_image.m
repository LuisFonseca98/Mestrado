%
% ISEL - Instituto Superior de Engenharia de Lisboa.
% 
% MEIC - Mestrado em Engenharia Informática e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria.
% 
% read_image.m
% Função que ilustra a manipulação de imagens com níveis de cinzento.
 
function read_image()  
  
% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc

%Octave
%pkg load image

% Ler a imagem a partir do ficheiro.

% Binária
%I = imread('man_bw.tif');

% Níveis de cinzento
% I = imread('camera.gif');
%I = imread('bird.gif');
%I = imread('lena.gif');
I = 0.5 * imread('lena.gif');
% I = imread('squares.gif');


% Obter as dimensões (resolução da imagem).
[M, N] = size(I); 
 
% Imprimir mensagem com as dimensões e resolução da imagem.
fprintf('Image with resolution %d x %d = %d pixels\n', M, N, M*N);

H = entropy(I)

% Lançar nova janela de figura e mostrar a imagem em níveis de cinzento
% e o respetivo histograma.
figure(1);
subplot(121); imshow(I); colorbar; title(' Image ' );
subplot(122); imhist(I); title( sprintf(' Histogram. H=%.2f\n',H) );
impixelinfo;

% Escrever a imagem em níveis de cinzento como ficheiro PNG.
imwrite( I, 'out.png' );

% Calcular a energia da imagem.
E = sum(sum( I.^2 ));

% Calcular a potência da imagem.
P = E / (M*N);

% Calcular o valor médio da imagem.
m = sum(sum( I )) / (M*N);

fprintf(' Energy=%d, Power=%d, Mean=%.2f\n', E, P, m);

% Calcular o valor mínimo e valor máximo da imagem.
mi  = min(min(I));
mx = max(max(I));
fprintf(' Min=%d, Max=%d \n', mi, mx);

% 
% 
% for level=0.1 : 0.01: 0.99
% %     
% % l = graythresh(I);
% % Ib = im2bw(I,0.1);
% % figure(2);
% % imshow(Ib);
% % impixelinfo;
% 
%     Ib = im2bw(I,level);
%     figure(3);
%     imshow(Ib); title(num2str(level));
%     impixelinfo;
%     pause(2);
% end

end

