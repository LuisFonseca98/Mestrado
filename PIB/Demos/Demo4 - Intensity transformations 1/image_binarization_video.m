%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Inform�tica e Multim�dia.
%  
% PIB - Processamento de Imagem e Biometria.
%
% image_binarization_video.m
%   

function image_binarization_video()

% Fechar todas as janelas de figuras.
close all;

% Ler a imagem com n�veis de cinzento.
I = imread( 'finger.tif' );

% Limiar de binariza��o
Th = 0.1 : 0.01 : 0.99;

% Criar o objeto para escrita do v�deo em MPEG-4
writerObj = VideoWriter  ('image_binarization.mp4',  'MPEG-4');

% Escolher 10 frames por segundo.
writerObj.FrameRate = 1;

% Abrir o objeto para gera��o das frames.
open(writerObj);

% Para todos os valores de Th...
frame_counter = 1;
for t = Th
    
    t
    frame_counter
    frame_counter = frame_counter + 1;
    % Binarizar com o limiar atual.
    IBW = im2bw(I, t);
    
    % Mostrar a imagem no mapa de cores de n�veis de cinzento.
    imagesc(IBW); colormap('gray');  title( sprintf('Th=%d ', t));
    
    % Obter uma frame a partir da imagem atual
    frame = getframe;
     
    % Escrever a frame no v�deo.
    writeVideo(writerObj,frame);
    
    pause(.5)
end
 
% Fechar o objeto de v�deo.
% Provoca a escrita no ficheiro.
close(writerObj);

