%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%  
% PIB - Processamento de Imagem e Biometria.
%
% DFT_properties.m
% Some DFT properties.
 
function DFT_properties()

close all

% Read test image
%f = imread('bird.gif');
f = imread('squares.gif');

% Resolution of the image
[M,N] = size(f);

% Module of the spectrum
F = fft2(f);
F = abs(F);

% Average on the spatial domain
ms = (1 / (M*N)) * sum(sum(double(f)))

% Average on the frequency domain
mf = (1 / (M*N)) * F(1,1)

% Power of the image - computed in the spatial domain
Ps = (1 / (M*N)) * sum(sum(double(f).^2))

% Center the spectrum
F = fftshift(F);

% Display the image
figure(1); 
subplot(121); imshow(f); title( 'f[m,n] ');
subplot(122); imagesc( log( 1 + F) ); 
colormap('gray'); title(' |F[u,v]| ');

% Power of the frequency domain
Pf = (1 / (M*N)^2) * sum(sum(F.^2))

F(M/2+1,N/2+1)
PDC  = (1 / (M*N)^2) * F(M/2+1,N/2+1)^2
pPDC = 100*PDC / Pf

end
