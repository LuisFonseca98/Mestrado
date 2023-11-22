% 
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
%
% test_fspecial.m
% Função que ilustra a utilização da função 'fspecial' para gerar
% máscaras para diversos filtros.

function test_fspecial() 

clc
close all;

% Filtro de média (quadrado)
% h = fspecial('average', hsize) returns an averaging filter h of size hsize. 
% The argument hsize can be a vector specifying the number of rows and 
% columns in h, or it can be a scalar, in which case h is a square matrix. 
% The default value for hsize is [3 3].
h = fspecial('average', [5,5])
figure(1); set(gcf,'Name', 'Average Filter');
surf(h); xlabel('x'); ylabel('y'); title('Average Filter');

% Filtro de média (circular)
% h = fspecial('disk', radius) returns a circular averaging filter (pillbox) within 
% the square matrix of size 2*radius+1. The default radius is 5.
h = fspecial('disk', 7)
figure(2); set(gcf,'Name', 'Pillbox Filter');
surf(h); xlabel('x'); ylabel('y'); title('Pillbox Filter');

% Filtro gaussiano
% h = fspecial('gaussian', hsize, sigma) returns a rotationally symmetric Gaussian 
% lowpass filter of size hsize with standard deviation sigma (positive). 
% hsize can be a vector specifying the number of rows and columns in h, 
% or it can be a scalar, in which case h is a square matrix. The default value 
% for hsize is [3 3]; the default value for sigma is 0.5.
h = fspecial('gaussian', 11, 1.5)
figure(3); set(gcf,'Name', 'Gaussian Filter');
surf(h); xlabel('x'); ylabel('y'); title('Gaussian Filter');

% Filtro Laplaciano
% h = fspecial('laplacian', alpha) returns a 3-by-3 filter approximating the 
% shape of the two-dimensional Laplacian operator. The parameter 
% alpha controls the shape of the Laplacian and must be in the range 
% 0.0 to 1.0. The default value for alpha is 0.2.
for alpha=0 : 0.1 : 1
    h = fspecial('laplacian', alpha)
    figure(4); set(gcf,'Name', 'Laplacian Filter');
    surf(h); xlabel('x'); ylabel('y'); title( sprintf('Laplacian, alpha= %.2f', alpha));
    pause(2);
end

% LoG
% h = fspecial('log', hsize, sigma) returns a rotationally symmetric 
% Laplacian of Gaussian filter of size hsize with standard deviation sigma (positive). 
% hsize can be a vector specifying the number of rows and columns in h, or 
% it can be a scalar, in which case h is a square matrix. The default value 
% for hsize is [5 5] and 0.5 for sigma.
for sigma=0.1 : 0.1 : 3
    h = fspecial('log', [7, 7], sigma)
    figure(5); set(gcf,'Name', 'LoG Filter');
    surf(h); xlabel('x'); ylabel('y'); title( sprintf('LoG, sigma= %.2f', sigma));
    pause(2);
end
