%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria
%
% my_webcam_v2.m  
% Using the Webcam to acquire images and perform a blur effect
%
function my_webcam_v2()

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
    Iblurred(:,:,1) = uint8(filter2(k,I(:,:,1)));
    Iblurred(:,:,2) = uint8(filter2(k,I(:,:,2)));
    Iblurred(:,:,3) = uint8(filter2(k,I(:,:,3)));
    
    figure(1); 
    subplot(211); imshow(I);               title( sprintf('Original %d',idx) );
    subplot(212); imshow(Iblurred);    title( sprintf('Blurred %d',idx) );
    
    %figure(2); imshow(img); title( sprintf('Imagem %d',idx) );
    %imwrite(img, sprintf('webcam_im%d.jpg',idx), 'jpg' );
    pause(2);
end

%preview(cam)