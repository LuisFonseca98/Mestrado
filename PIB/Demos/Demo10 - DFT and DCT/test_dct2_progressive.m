%
% ISEL - Instituto Superior de Engenharia de Lisboa.
%
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%   
% PIB - Processamento de Imagem e Biometria.
%
% test_dct2_progressive.m
% Analysis of the DCT2D. 

function test_dct2_progressive()

% Clear console.
clc

% Close all windows
close all

% Input image
%C:\Program Files\MATLAB\R2014a\toolbox\images\imdata
%f = imread('squares.gif');
f = imread('bird.gif');
%f = imread('circuit.tif');

% Energy of the image - spatial domain
Ef = sum(sum(f.^2))

% MSE and energy vectors.
mse = zeros(1,64);
en  = zeros(1,64);

% Increase on the number of levels.
for level = 1 : 64
 
    % DCT2 in 8x8 blocks.
    % Define the function for each block, with
    % the creesponding level.
    fun = @(block_struct) T1(block_struct.data,level);
    
    % Process block by block.
    F = uint8(blockproc(f, [8,8], fun ));
    
    % Energy of the reconstructed image 
    en(level) = sum(sum( abs(F).^2));
    % Compute the MSE between the original and the reconstructed image
    mse(level) = sum(sum(f-F).^2) / numel(f);   
    
    % Image and reconstruction with level coefficients
    figure(1);
    subplot(121); imagesc(f); title('X'); colormap('gray'); 
    subplot(122); imagesc(F); title( ['Level= ' num2str(level) '/64 '] );
    xlabel( sprintf('MSE=%.2f', mse(level)) );
        
    %pause(1)
    filename = ['level_' num2str(level) '.png'];
    print('-dpng',filename);
end

% Show the MSE between the original 
% and the reconstructd image.
figure(2);
plot(mse, 'Linewidth',2); grid on; 
title('MSE - Mean Squared Error (f,F)');
ylabel('MSE'); xlabel('# DCT coefficients');
print '-dpng' 'fig2.png' 

% Show the energy of the reconstructed image
% of the number of levels
en = 100*en / Ef;
figure(3);
plot(en, 'Linewidth',2); grid on; 
title('Percentage of energy of the reconstructed image');
ylabel('Energy'); xlabel('# DCT coefficients');
print '-dpng' 'fig3.png' 
 
end


function y = T1(x, level)

% Compute the DCT for the 8x8 block
y = dct2(x);

% Keep only 'level' coefficients
% Set to zero all the remaining coefficients
if level < numel(y)
    y ( level+1 : end) = 0;
end

% Perform the IDCT
y = idct2(y);
end
