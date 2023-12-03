%
% ISEL - Instituto Superior de Engenharia de Lisboa.
% 
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%  
% PIB - Processamento de Imagem e Biometria.
%  
% DCT_properties.m
% Some DCT properties.

function DCT_properties()

close all

% Read test image
%f = imread('bird.gif');
%f = imread('squares.gif');
f = imread('woman.tif');
%f = imread('rectangle.tif');

% Resolution of the image
[M,N] = size(f);

% The DCT of the image
F = dct2(f);

% Average of the image - computed on the spatial domain
ms = (1 / (M*N)) * sum(sum(double(f)))

% Average of the image - computed on the DCT domain
md = (1 / (sqrt(M)*sqrt(N)))* F(1,1)

% Power of the image - computed on the spatial domain
Ps = (1 / (M*N)) * sum(sum(double(f).^2))
 
% Display the image
figure(1); 
subplot(121); imagesc(f); title( 'f[m,n] '); axis off;
subplot(122); imagesc( log(1+abs(F)) ); colormap('gray'); 
title(' F[u,v] (DCT) '); axis off;

% Power of the image - computed on the DCT domain
Pd = (1/ (M*N)) * sum(sum(F.^2))
 
% The (0,0) coefficient
F_00 = F(1,1)

% The DC power
PDC  = (1 / (M*N)) * F(1,1)^2

% The percentage of DC power
pPDC = 100*PDC / Pd
 
print '-dpng' 'dct.png'
end
