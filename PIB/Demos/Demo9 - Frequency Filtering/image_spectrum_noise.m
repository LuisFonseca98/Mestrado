%
% ISEL - Instituto Superior de Engenharia de Lisboa.
% 
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
%
% image_spectrum.m
% Spectrum (module and phase) of some test images.
  
function image_spectrum_noise()

close all

M = 256; 
N = 256;
noise1 = zeros( M, N );
for m=1 : M
    for n=1 : N
        noise1(m,n) = 180*cos( (pi/4)*m ) * cos( (pi/32)*n );   
    end
end
noise1 = uint8(noise1');

% Compute the DFT
F = fft2(noise1);

% Center the spectrum
F = fftshift(F);

% Display the module of the spectrum 
figure(1); 
subplot(121); imagesc(noise1); title(' f[m,n] ');
axis off; colorbar;
subplot(122); imagesc( log( 1 + abs(F)) ); 
colormap('gray'); title(' |F[u,v]| '); axis off; colorbar;
print '-dpng' 'fig_noise.png'

% % Display the angle of the spectrum 
% subplot(122); imagesc( angle(F) ); 
% colormap('gray'); title(' arg[F[u,v]] ');


end
