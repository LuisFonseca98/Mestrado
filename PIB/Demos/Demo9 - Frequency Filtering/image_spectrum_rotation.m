%
% ISEL - Instituto Superior de Engenharia de Lisboa.
% 
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
%
% image_spectrum_rotation.m
% Analysis of the rotation effect on the spectrum.
   
function image_spectrum_rotation()

close all

% Read test image
%I = imread('bird.gif');
%I = imread('squares.gif');
%I = imread('circles.bmp');
%I = imread('rectangle.tif');
I = imread('woman.tif');

% Zero-padded image to resolution P=2M and Q=2N.
Ip = [ I, zeros(size(I,1), size(I,2));
       zeros(size(I,1), size(I,2)), zeros(size(I,1), size(I,2))];
Ip = uint8(Ip);

% Compute the DFT
F = fft2(Ip);

% Center the spectrum
F = fftshift(F);

% Display the image
figure(1); 
imshow(I); title( 'f[m,n] ');

figure(2);
% Display the module of the spectrum 
subplot(121); imagesc( log( 1 + abs(F)) ); 
colormap('gray'); title(' |F[u,v]| ');
% Display the angle of the spectrum 
subplot(122); imagesc( angle(F) ); 
colormap('gray'); title(' arg[F[u,v]] ');

% Perform several rotations an check the spectrum
for ang = -90 : 5 : 90
    Ir = imrotate(I, ang );
    % Zero-padded image to resolution P=2M and Q=2N.
    Irp = [ Ir, zeros(size(Ir,1), size(Ir,2));
           zeros(size(Ir,1), size(Ir,2)), zeros(size(Ir,1), size(Ir,2))];
    Irp = uint8(Irp);

    Fr = fft2(Irp);
    Fr = fftshift(Fr);

    figure(1); 
    subplot(131); imshow(Ir); title( [ 'f[m,n] ' num2str(ang)] );
    subplot(132); imagesc( log( 1 + abs(Fr)) ); 
    colormap('gray'); title(' |F[u,v]| ');
    subplot(133); imagesc( angle(Fr) ); 
    colormap('gray'); title(' arg[F[u,v]] ');
    pause(1);
end
end
