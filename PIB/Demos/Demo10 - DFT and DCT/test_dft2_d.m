%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%  
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
% 
% PIB - Processamento de Imagem e Biometria.
%  
% test_dft2_d.
% Análise de resultados da DFT2D. 

function test_dft2_d()

% Limpar a consola.
clc

% Imagem de teste.
f = [   
        1  1  1  1  4 ;
        1  1  1  1  5 ;
        1  1  1  1  6 ;
        1  1  1  1  7 ;
        4  5  6  7  7 ];

% A DFT 2D
Y1 = fft2(f)

% Ponto u=v=0
Y1(1,1)
sum(sum(f))

% Espetro não centrado 
Y1

% Espetro centrado 
Y1 = fftshift(Y1)

Ex1 = sum(sum(f.^2))

Y1 = abs(Y1);
Ey1 = sum(sum(Y1.^2)) / (size(Y1,1) * size(Y1,2))

