function exe1 ()
clc 

F = [ 18,    -2-j*2,       2,       -2+j*2;
          2,       0,          -2,          0;
         -2,     2-j*2,       -2          2 + j*2;
         2,        0,           -2,        0];

Fcentered = fftshift(F)

mod_F = abs(Fcentered)

angle_F = angle(Fcentered)
