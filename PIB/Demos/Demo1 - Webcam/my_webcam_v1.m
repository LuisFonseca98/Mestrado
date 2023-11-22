%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria
%
% my_webcam_v1.m  
% Using the Webcam to acquire images and deal with color
%
 function my_webcam_v1()

%https://www.mathworks.com/videos/webcam-support-89504.html
%https://www.mathworks.com/help/supportpkg/usbwebcams/ug/acquire-images-from-webcams.html
list = webcamlist

cam = webcam('HD')

% Video preview
% preview(cam)

NI = 10;

for idx = 1:NI
    
    fprintf('Imagem %d de %d \n', idx, NI); 
    I = snapshot(cam);
    In = 255 - I;
    Ig = rgb2gray(I);
    
    [M, N, B] = size(I);
    
    IR = zeros( M, N, B) ;
    IG = zeros( M, N, B) ;
    IB = zeros( M, N, B) ;
    
    IR(:,:,1)= Ig;  
    IR = uint8(IR);
    IG(:,:,2)= Ig;  
    IG = uint8(IG);
    IB(:,:,3)= Ig;  
    IB = uint8(IB);
        
    figure(1); 
    subplot(231); imshow(I);      title( sprintf('Original %d',idx) );
    subplot(232); imshow(IR);    title( sprintf('Red %d',idx) );
    subplot(233); imshow(IG);    title( sprintf('Green %d',idx) );
    subplot(234); imshow(IB);    title( sprintf('Blue %d',idx) );
    subplot(235); imshow(In);     title( sprintf('Negative %d',idx) ); 
    subplot(236); imshow(Ig);     title( sprintf('Grayscale %d',idx) );
    
    %figure(2); imshow(img); title( sprintf('Imagem %d',idx) );
    %imwrite(img, sprintf('webcam_im%d.jpg',idx), 'jpg' );
    pause(2);
end

%preview(cam)