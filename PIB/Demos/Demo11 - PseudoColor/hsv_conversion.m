%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%   
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
%  
% hsv_conversion.m
% Análise de operações em espaços de cor. Modificação das
% componentes H, S e V e análise do efeito no espaço RGB.
%

function hsv_conversion()

% Fechar todas as janelas com figuras.
close all;

% Tempo de pausa entre imagens.
T = 1.5;
 
% Ler a imagem RGB para uma matriz (M x N x 3).
%I = 1.2*imread('monarch.tif');
I = 1.2*imread('peppers.png');

% Converter de RGB para HSV
I_HSV = rgb2hsv(I);

% Estabelecer/modificar o valor de H.
I_HSV(:,:,1) = 0;

% Converter de HSV para RGB.
I_RGB = hsv2rgb(I_HSV);

imshowpair(I, I_RGB,'montage')

% pause

for h=0.0 : 0.01 : 0.99

    % Estabelecer/modificar o valor de H.
    I_HSV(:,:,1) = h;

    % Converter de HSV para RGB.
    I_RGB = hsv2rgb(I_HSV);

    imshowpair(I, I_RGB,'montage')
    title(sprintf('H=%.2f', h));
    pause(.5)
end


return


