%
% ISEL - Instituto Superior de Engenharia de Lisboa.
% 
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%  
% PIB - Processamento de Imagem e Biometria.
%
% optical_ilusion2.m
% Função que mostra um efeito de ilusão ótica do sistema visual humano.
%
function optical_ilusion2() 

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc

blk = 100*ones(64,64);
  
I1  = 10 + zeros(256, 256);
I1( 96:159, 96:159 ) = blk;
I1 = uint8(I1);

I2  = 128 + zeros(256, 256);
I2( 96:159, 96:159 ) = blk;
I2 = uint8(I2);

I3  = 250 + zeros(256, 256);
I3( 96:159, 96:159 ) = blk;
I3 = uint8(I3);

figure(1);
subplot(131); imshow(I1); colormap('gray'); title('A');
subplot(132); imshow(I2); colormap('gray'); title('B');
subplot(133); imshow(I3); colormap('gray'); title('C');
impixelinfo;

figure(2);
subplot(231); imshow(I1); colormap('gray'); title('A');
subplot(234); imhist(I1); colormap('gray'); title('A');
subplot(232); imshow(I2); colormap('gray'); title('B');
subplot(235); imhist(I2); colormap('gray'); title('B');
subplot(233); imshow(I3); colormap('gray'); title('C');
subplot(236); imhist(I3); colormap('gray'); title('C');
impixelinfo;

H1 = entropy(I1)
H2 = entropy(I2)
H3 = entropy(I3)

m1 = mean(mean(I1))
m2 = mean(mean(I2))
m3 = mean(mean(I3))

end


