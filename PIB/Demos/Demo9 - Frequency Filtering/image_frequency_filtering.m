%
% ISEL - Instituto Superior de Engenharia de Lisboa.
% 
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria.
%
% image_frequency_filtering.m
% Frequency filtering of images.
 
function image_frequency_filtering()
 
close all 

% Read test image
I = imread('bird.gif');
%I = imread('squares.gif');

% Apply a low-pass filter with different radius
for r = 10 : 10 : 140
    frequency_filtering_low_pass( I, r );
    pause(5);
end

% Apply a high-pass filter with different radius
for r = 5 : 5 : 35
    frequency_filtering_high_pass( I, r );
    pause(5);
end

end

function If = frequency_filtering_low_pass( I, radius )

% Steps 1 and 2  - Zero-padded image to resolution P=2M and Q=2N.
Ip = [ I, zeros(size(I,1), size(I,2));
       zeros(size(I,1), size(I,2)), zeros(size(I,1), size(I,2))];
Ip = uint8(Ip);
P = size(Ip,1);
Q = size(Ip,2);

% Steps 3 and 4 - Compute the DFT of the padded image and center its spectrum.
Fp = fftshift(fft2(Ip));

% Step 5 - Set the filter mask H, on the frequency domain.
% In this case, it is a low-pass filter.
H = zeros( P, Q );
for m=1 : P
    for n=1 : Q
        d = sqrt( (m - P/2)^2 + (n - Q/2)^2);
        if d < radius
            H(m,n)=1;
        end
    end
end
% Multiply the spectrum of the padded image
% with the filter mask H.
G = Fp .* H;

% Step 6 - Perform the inverse DFT (with centered spectrum).
% Remove residual imaginary parts and convert to unsigned integer.
If = ifft2 ( ifftshift(G) ) ;
If = uint8(real(If));

% Step 7 - Crop the result
If = uint8(If( 1:size(I,1), 1:size(I,2)));

% Analysis of the main steps of the algorithm.
figure(1); 
subplot(221); imshow(Ip); title(' Padded image');
subplot(222); imagesc(abs(Fp)); 
title(' Spectrum of the padded image');
colormap('gray');
subplot(223); imshow(H); title(' Spectrum of H ');
subplot(224); imshow(If); title(' Padded filtered image');

figure(2); 
subplot(121); imshow(I);  title(' Original ');
subplot(122); imshow(If); title( sprintf(' Filtered (radius=%d)',radius) );
end



function If = frequency_filtering_high_pass( I, radius )

% Step 1 - Zero-padded image to resolution P=2M and Q=2N.
Ip = [ I, zeros(size(I,1), size(I,2));
       zeros(size(I,1), size(I,2)), zeros(size(I,1), size(I,2))];
Ip = uint8(Ip);
P = size(Ip,1);
Q = size(Ip,2);

% Step 2 - Compute the DFT of the padded image and center its spectrum.
Fp = fftshift(fft2(Ip));

% Step 3 - Set the filter mask H, on the frequency domain.
% In this case, it is a high-pass filter.
H = zeros( P, Q );
for m=1 : P
    for n=1 : Q
        d = sqrt( (m - P/2)^2 + (n - Q/2)^2);
        if d > radius
            H(m,n)=1;
        end
    end
end

% Step 4 - Multiply the spectrum of the padded image
% with the filter mask H.
G = Fp .* H;

% Steps 5 and 6 - Perform the inverse DFT (with centered spectrum).
% Remove residual imaginary parts and convert to unsigned integer.
If = ifft2 ( ifftshift(G) ) ;
If = uint8(real(If));

% Step 7 - Crop the resukting image
If = uint8(If( 1:size(I,1), 1:size(I,2)));

% Analysis of the main steps of the algorithm.
figure(1); 
subplot(221); imshow(Ip); title(' Padded image');
subplot(222); imagesc(abs(Fp)); title(' Spectrum of the padded image');
subplot(223); imshow(H); title(' Spectrum of H ');
subplot(224); imshow(If); title(' Padded filtered image');

figure(2); 
subplot(121); imshow(I);  title(' Original ');
subplot(122); imshow(If); title( sprintf(' Filtered (radius=%d)',radius) );
end


