function exe4 ()

clc 

G = [ 9,   0,   0;
       -2,    0,   0;
       0,     0,   0
      ];

E_G = sum(sum(G.^2))

g = idct2( G )
S = sum(sum(g))
E_g = sum(sum(g.^2))