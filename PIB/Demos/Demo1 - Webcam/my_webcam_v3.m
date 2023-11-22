%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria
%
% my_webcam_v3.m  
% Using the Webcam to acquire images and perform a blur effect
%
function my_webcam_v3()

%https://www.mathworks.com/videos/webcam-support-89504.html
%https://www.mathworks.com/help/supportpkg/usbwebcams/ug/acquire-images-from-webcams.html
list = webcamlist

cam = webcam('HD User Facing')

% Video preview
% preview(cam)

M = 35;
k = (1/M^2) * ones(M,M);

NI = 10;

for idx = 1:NI
    
    fprintf('Imagem %d de %d \n', idx, NI); 
    I = snapshot(cam);
   
    [M, N, B] = size(I);
    Iblurred = zeros( 736, N, B) ;
    
    Iblurred(:,:,1) = blockproc(I(:,:,1), [32, 32], @process_block);
    Iblurred(:,:,2) = blockproc(I(:,:,2), [32, 32], @process_block);
    Iblurred(:,:,3) = blockproc(I(:,:,3), [32, 32], @process_block);
    
    Iblurred(:,:,1) = uint8(filter2(k,Iblurred(:,:,1)));
    Iblurred(:,:,2) = uint8(filter2(k,Iblurred(:,:,2)));
    Iblurred(:,:,3) = uint8(filter2(k,Iblurred(:,:,3)));
    
    Iblurred = uint8( Iblurred );
    
    figure(1); 
    subplot(211); imshow(I);                 title( sprintf('Original %d',idx) );
    subplot(212); imshow(Iblurred);     title( sprintf('Blurred %d',idx) );
    
    %figure(2); imshow(img); title( sprintf('Imagem %d',idx) );
    %imwrite(img, sprintf('webcam_im%d.jpg',idx), 'jpg' );
    pause(2);
end
end

%preview(cam)

function Y = process_block(X)
n = 2;
Z = dct2(X.data);
W = zeros(32, 32);
W(1:n,1:n) = Z(1:n, 1:n);
Y= idct2(W);
end

