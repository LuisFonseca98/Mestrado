function exe1()

G = [9 0 0; -2 0 0; 0 0 0]
 
g = idct2(G)

Sg = sum(sum(g))

EG = sum(sum(G.^2))
Eg = sum(sum(g.^2))
end

