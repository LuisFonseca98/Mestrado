%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%  
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
% 
% test_dct2_energy.m
% Analysis of the DCT2D energy preservation property. 
  
function test_dct2_energy()

% Clear console.
clc

% Close all windows
close all

% Input image 
%f = magic(10)
f = 255*rand(128,128);

% Energy of the image - spatial domain
Ef = sum(sum(f.^2))
  
% Compute the DCT 
F = dct2(f); 

% Energy of the image - DCT domain
EF = sum(sum(F.^2))

% Check for the percentage of energy, contained in the DCT
% coefficients.
perc = 100*EF / Ef

return
