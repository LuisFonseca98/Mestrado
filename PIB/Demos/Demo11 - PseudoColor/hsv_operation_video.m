%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%  
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria.
%  
% hsv_operation_video.m
% Análise de operações em espaços de cor. Modificação das
% componentes H, S e v e análise do efeito no espaço RGB.
%

function hsv_operation_video()

% Fechar todas as janelas com figuras.
close all;

% Tempo de pausa entre imagens.
T = 2.5;

% Ler a imagem RGB para uma matriz (M x N x 3).
%I = imread('monarch.tif');
I = imread('peppers.png');

% Criar o objeto para escrita do vídeo em MPEG-4
writerObj = VideoWriter  ('hsv_video_image.mp4',  'MPEG-4');

% Escolher 4 frames por segundo.
writerObj.FrameRate = 4;

% Abrir o objeto para geração das frames.
open(writerObj);

% Converter de RGB para HSV
I_HSV = rgb2hsv(I);

% Modificar os valores de H de forma sequencial.
% Todos os pixels ficam com o mesmo valor de H.
figure(2);
for h= 0.0 : 0.05 : 0.95
    
    % Estabelecer/modificar o valor de H.
    I_HSV(:,:,1) = h;
    
    % Converter de HSV para RGB.
    I_RGB = hsv2rgb(I_HSV);
    
    % Mostrar a imagem resultante durante T segundos.
    subplot(121); imshow(I); title('Original RGB');
    subplot(122); imshow(I_RGB); title( sprintf('H= %.2f',h) );
    pause(T);
    
    % Obter uma frame a partir da imagem atual
    frame = getframe;
    
    % Escrever a frame no vídeo.
    writeVideo(writerObj,frame);
end

% Converter de RGB para HSV
I_HSV = rgb2hsv(I);

% Modificar os valores de S de forma sequencial.
% Todos os pixels ficam com o mesmo valor de S.
figure(2);
for s= 0.0 : 0.05 : 0.95
    
    % Estabelecer/modificar o valor de S.
    I_HSV(:,:,2) = s;
    
    % Converter de HSV para RGB.
    I_RGB = hsv2rgb(I_HSV);
    
    % Mostrar a imagem resultante durante 2 segundos.
    subplot(121); imshow(I); title('Original RGB');
    subplot(122); imshow(I_RGB); title( sprintf('S= %.2f',s) );
    pause(T);
    
    % Obter uma frame a partir da imagem atual
    frame = getframe;
    
    % Escrever a frame no vídeo.
    writeVideo(writerObj,frame);
    
end

% Converter de RGB para HSV
I_HSV = rgb2hsv(I);

% Modificar os valores de V de forma sequencial.
% Todos os pixels ficam com o mesmo valor de V.
figure(2);
for v= 0.0 : 0.05 : 0.95
    
    % Estabelecer/modificar o valor de V.
    I_HSV(:,:,3) = v;
    
    % Converter de HSV para RGB.
    I_RGB = hsv2rgb(I_HSV);
    
    % Mostrar a imagem resultante durante 2 segundos.
    subplot(121); imshow(I); title('Original RGB');
    subplot(122); imshow(I_RGB); title( sprintf('I= %.2f',v) );
    pause(T);
     
    % Obter uma frame a partir da imagem atual
    frame = getframe;
     
    % Escrever a frame no vídeo.
    writeVideo(writerObj,frame);
end

% Fechar o objeto de vídeo.
% Escrita no ficheiro.
close(writerObj);

return


