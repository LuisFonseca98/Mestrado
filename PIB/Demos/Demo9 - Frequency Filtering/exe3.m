function exe3 ()
 
clc

F = [ 24,                 3+j*sqrt(3),       3-j*sqrt(3);
       -3+j*sqrt(3),     -3-j*sqrt(3),        0;
       -3-j*sqrt(3),          0,                -3+j*sqrt(3)
       ];

m_f = (1/9) * F(1,1)
abs_F = abs(F);
P_f = (1/9)^2 * sum(sum(abs_F.^2))

f = ifft2( F )
m_f_space = mean(mean(f))
P_f_space = (1/9) * sum(sum(f.^2))
