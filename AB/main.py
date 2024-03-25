import newick
from pylocluster import *

matrix = squareform([0.5,0.67,0.8,0.2,0.4,0.7,0.6,0.8,0.8,0.3])
nwk = linkage(matrix, taxa=['G', 'S', 'I', 'E', 'D'], method='upgma')
tree = newick.loads(nwk)[0]
print(tree.ascii_art())


