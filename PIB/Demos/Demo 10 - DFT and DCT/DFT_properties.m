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
%f = imread('squares.gif');
f = imread('woman.tif');
%f = imread('rectangle.tif');


% Resolution of the image
[M,N] = size(f);

% Compute the DFT - the spectrum of the image
F = fft2(f);

% Average of the image - computed on the spatial domain
ms = (1 / (M*N)) * sum(sum(double(f)))

% Average of the image - computed on the frequency domain
mf = (1 / (M*N)) * F(1,1)

% Power of the image - computed on the spatial domain
Ps = (1 / (M*N)) * sum(sum(double(f).^2))

% Center the spectrum
F = fftshift(F);

% Compute the module of the spectrum
Fabs = abs(F);

% Display the image
figure(1); 
subplot(131); imagesc(f); title( 'f[m,n] ');
axis off;
subplot(132); imagesc( log(1 + Fabs) ); colormap('gray'); title(' |F[u,v]| ');
%subplot(232); imagesc( Fabs ); colormap('gray'); title(' |F[u,v]| ');
axis off;
subplot(133); imagesc( angle(F) ); colormap('gray'); title(' arg[F[u,v]] ');
axis off;

% Power of the image - computed on the frequency domain
Pf = (1 / (M*N)^2) * sum(sum(Fabs.^2))
 
% The (0,0) coefficient - module
F_00 = Fabs(M/2+1,N/2+1)

% The DC power
PDC  = (1 / (M*N)^2) * Fabs(M/2+1,N/2+1)^2
 
% The percentage of DC power
pPDC = 100*PDC / Pf

print '-dpng' 'spectrum.png'

end
