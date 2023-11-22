%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Inform�tica e Multim�dia.
%  
% PIB - Processamento de Imagem e Biometria.
%
% image_transition_video.m
%   
 
function image_transition_video()

% Fechar todas as janelas de figuras.
close all;

% Ler as duas imagens com n�veis de cinzento.
I1 = imread( 'bird.gif' );
I2 = imread( 'camera.gif' );

% Valores poss�veis para o fator \alpha
alfa = 0 : 0.0125 : 1;

% Criar o objeto para escrita do v�deo em MPEG-4
writerObj = VideoWriter  ('image_sum.mp4',  'MPEG-4');

% Escolher 10 frames por segundo.
writerObj.FrameRate = 1;

% Abrir o objeto para gera��o das frames.
open(writerObj);


% Valores poss�veis para o fator \alpha
alfa = 0 : 0.05 : 1;

% Para todos os valores de \alpha...
for a = alfa
    
    % Criar a imagem por combina��o linear das duas imagens.
    I = a*I1+ (1-a)*I2;
    
    % Mostrar a imagem no mapa de cores de n�veis de cinzento.
    imagesc(I); colormap('gray');  title( sprintf('%.2f I_1 + %.2f I_2 ', a, 1-a));
    
    % Obter uma frame a partir da imagem atual
    frame = getframe;
    
    % Escrever a frame no v�deo.
    writeVideo(writerObj,frame);
    
    pause(.5)
end

% Fechar o objeto de v�deo.
% Provoca a escrita no ficheiro.
close(writerObj);

