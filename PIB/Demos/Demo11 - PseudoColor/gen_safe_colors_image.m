%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%   
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
%  
% gen_safe_colors_image.m
% Geração das cores seguras (safe colors) em RGB. Análise dos
% valores das respetivas componentes H, S e v 
%
 
function gen_safe_colors_image()

% Fechar todas as janelas com figuras.
close all;

% Tempo de pausa entre imagens.
T = 0.05;

% Os valores das componentes das 'safe colors'.
safe_values = 0 : 51 : 255;

% Criar a imagem RGB de resolução 16 x 16 a zeros.
I = zeros(16, 16, 3);

% Criar a imagem final.
Is = zeros( 288, 192, 3);

figure(1);
row = 1;
col = 1;
% Produzir as 216 = 6^3 cores distintas.
for r = safe_values
    for g = safe_values
        for b = safe_values
            
            % Título da figura.
            curr_color = ['[' num2str(r) ',' num2str(g) ',' num2str(b) ']'];
            
            % Afetar as componentes de cor em RGB.
            I(:,:,1) = r;  I(:,:,2) = g; I(:,:,3) = b;
            
            % Obter a imagem RGB.
            I_RGB = uint8(I);
            
            % Converter para HSV.
            I_HSV = rgb2hsv(I_RGB);
            
            % Mostrar as quatro imagens: RGB, H, S e V.
            subplot(221); imshow(I_RGB); title(['[R,G,B] =' curr_color]);
            subplot(222); imshow(I_HSV(:,:,1)); 
            title( ['H=' num2str(max(max(I_HSV(:,:,1))))] );
            subplot(223); imshow(I_HSV(:,:,2)); title('S');
            title( ['S=' num2str(max(max(I_HSV(:,:,2))))] );
            subplot(224); imshow(I_HSV(:,:,3)); title('V');
            title( ['V=' num2str(max(max(I_HSV(:,:,3))))] );
            pause(T);
            
            Is(row:row+15, col:col+15,1) = I_RGB(:,:,1);
            Is(row:row+15, col:col+15,2) = I_RGB(:,:,2);
            Is(row:row+15, col:col+15,3) = I_RGB(:,:,3);
            
            row = row + 16;
            if row >= 288
                row=1;
                col = col + 16;
                if col >= 192
                    col=1;
                end
            end
        end
    end
end

figure(2);
imshow(uint8(Is)); impixelinfo;
title(' 216 RGB safe colors ');
imwrite(uint8(Is),'safe-colors.png','png');

return