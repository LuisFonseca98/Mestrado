function exe2 ()
clc
 
F = [ 13,    -1,       5,       -1;
          -j,     j,          -j,       j;
         -5,     1,       3          1;
         j,        -j,      j,        -j];

m_f = (1/16) * F(1,1)
abs_F = abs(F);
Pf = (1/16)^2 * sum(sum(abs_F.^2))

Fcentered = fftshift(F)
mod_F = abs(Fcentered)
angle_F = angle(Fcentered)


f = ifft2( F )
m_f_space = mean(mean(f))
P_f_space = (1/16) * sum(sum(f.^2))
