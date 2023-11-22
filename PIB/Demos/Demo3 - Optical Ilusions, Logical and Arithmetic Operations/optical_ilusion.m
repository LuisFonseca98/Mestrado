%
% ISEL - Instituto Superior de Engenharia de Lisboa.
% 
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%  
% PIB - Processamento de Imagem e Biometria.
%
% optical_ilusion.m
% Função que mostra um efeito de ilusão ótica do sistema visual humano.
%
function optical_ilusion() 

% Fechar todas as janelas de figuras.
close all;

% Limpar a consola.
clc 
 
% Ler a imagem a partir do ficheiro.
I = imread('square.png');
I = rgb2gray(I);
imshow(I);
impixelinfo; 

end

