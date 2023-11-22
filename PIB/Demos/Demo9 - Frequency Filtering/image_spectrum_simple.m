%
% ISEL - Instituto Superior de Engenharia de Lisboa.
% 
% MEIC - Mestrado em Engenharia Informatica e de Computadores.
% MEIM - Mestrado em Engenharia Informática e Multimédia.
%
% PIB - Processamento de Imagem e Biometria.
% 
% image_spectrum_simple.m
% Analysis of the spectrum of some simple test images.
 
function image_spectrum_simple()
 
close all;

% Resolução da imagem
M = 32;

% Imagem constante (DC)
I1 = 100 * ones(M,M);
display_image(I1);
print '-dpng' 'fig1.png'

% Imagem com linhas sem variação
I2 = [ 
    100 * ones(1,M);
    100 * ones(1,M);
      20 * ones(1,M);
      20 * ones(1,M)];
I2 = repmat(I2,8,1);    
display_image(I2);
print '-dpng' 'fig2.png'

% Imagem com colunas sem variação
I3 = I2';
display_image(I3);
print '-dpng' 'fig3.png'

% Imagem com variações horizontais e verticais.
I4  = I2 + I3;
display_image(I4);
print '-dpng' 'fig4.png'

% Imagem com variações horizontais e verticais e média nula.
I5 = I4 - mean(mean(I4));
display_image(I5);
print '-dpng' 'fig5.png'

% Imagem com diferentes largura de barra de níveis de cinzento.
I6 = [ 
    100 * ones(1,M);
    100 * ones(1,M);
    100 * ones(1,M);
    100 * ones(1,M);
    100 * ones(1,M);
    100 * ones(1,M);
    20 * ones(1,M);
    20 * ones(1,M)]
I6 = repmat(I6,4,1);    
display_image(I6);
print '-dpng' 'fig6.png'

% Imagem com variações horizontais e verticais.
I7 = I6 + I6';
display_image(I7);
print '-dpng' 'fig7.png'
end

function display_image(I)

figure;

% Calcular a DFT
F = fft2(I);

% Centrar o espetro
F = fftshift(F);

% Apresentar as imagem no domínio do espaço.
subplot(121); imagesc(I); 
colormap('gray'); title(' f[m,n] ');
colorbar
axis off;
%impixelinfo;

% Apresentar o módulo do espetro.
subplot(122); imagesc( angle(F) ); 
colormap('gray'); title( ' arg[F[u,v]] '); 
colorbar;
axis off;

end


function display_image2(I)

figure;

% Calcular a DFT
F = fft2(I);

% Centrar o espetro
F = fftshift(F);

% Apresentar as imagem no domínio do espaço.
subplot(131); imagesc(I); 
colormap('gray'); title(' f[m,n] ');
colorbar
axis off;
impixelinfo

% Apresentar o módulo do espetro.
subplot(132); imagesc( log( 1 + abs(F)) ); 
colormap('gray'); title( ' |F[u,v]| '); 
colorbar;
axis off; 

% Apresentar o argumento do espetro.
subplot(133); imagesc( angle(F) ); 
colormap('gray'); title( ' arg[F[u,v]] ');
colorbar;
axis off;
end