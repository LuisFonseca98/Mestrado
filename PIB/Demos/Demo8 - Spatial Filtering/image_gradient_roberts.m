% 
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria.
%  
% image_gradient_roberts.m
% Função que ilustra a aplicação do método de gradiente (Roberts).
 
function image_gradient_roberts()

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc

% Ler a imagem a partir do ficheiro.
%I = imread('circuit.tif');
I = imread('spine.tif');
%I = imread('squares.gif');

Ir = edge(I,'roberts');

% Comparação dos resultados com a imagem original.
% Imagens.
figure(1); set(gcf,'Name', 'Roberts');
subplot(121);  imshow(I); title(' Original' ); colormap('gray'); 
subplot(122);  imshow(Ir); axis tight; title(' Roberts' ); 

print '-dpng' 'roberts1.png'

wx = [ -1 0; 0 1 ];
Ir1 = filter2(wx,I);
wy = [ 0 -1; 1 0 ];
Ir2 = filter2(wy,I);
figure(2); set(gcf,'Name', 'Roberts');
subplot(121);  imagesc(uint8(abs(Ir1))); 
axis off; title(' Gx' ); colormap('gray');
subplot(122);  imagesc(uint8(abs(Ir2))); 
axis off; title(' Gy' ); colormap('gray'); 
print '-dpng' 'roberts2.png'
end



