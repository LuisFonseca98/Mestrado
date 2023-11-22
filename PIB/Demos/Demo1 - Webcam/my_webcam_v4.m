%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria
%
% my_webcam_v4.m  
% Using the Webcam to acquire images and perform face detection
%
function my_webcam_v4()

%https://www.mathworks.com/videos/webcam-support-89504.html
%https://www.mathworks.com/help/supportpkg/usbwebcams/ug/acquire-images-from-webcams.html
list = webcamlist

cam = webcam('HD User Facing')

% Video preview
% preview(cam)

faceDetector = vision.CascadeObjectDetector;

NI = 10;

for idx = 1:NI
    
    fprintf('Imagem %d de %d \n', idx, NI); 
    I = snapshot(cam);
   
 	bboxes = faceDetector(I);
    
    IFaces = insertObjectAnnotation(I,'rectangle',bboxes,'Face');   

    figure(1); 
    subplot(211); imshow(I);                 title( sprintf('Original %d',idx) );
    subplot(212); imshow(IFaces);      title( sprintf('Detected faces %d',idx) );
    
    %figure(2); imshow(img); title( sprintf('Imagem %d',idx) );
    %imwrite(img, sprintf('webcam_im%d.jpg',idx), 'jpg' );
    pause(2);
end
end

%preview(cam)
