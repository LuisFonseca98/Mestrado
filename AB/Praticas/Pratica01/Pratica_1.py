"""
Replication begins in a genomic region called the replication origin (denoted ori)
We assume that a genome has a single ori and is represented as a DNA string, or a string of nucleotides from the four-letter
alphabet {A, C, G, T}

A k-mer is a substring of length k contained within a biological sequence
The k-mer composition of a string str, denoted Compositionk(str), is the collection of all k-mer substrings of str (including repeats).
"""


def built_ori_from_string(input, k_mer=3):
    triplets = []
    for char in range(len(input) - 2):
        triplet = input[char:char + k_mer]
        triplets.append(triplet)
    return triplets

"""
Builts an bruijn graph passing a genome
"""


def bruijn_graph(k_meters):
    graphs = {}  # create a empty list of graphs
    for meters in k_meters:  # loop over each k-mer in the input list of k-mers.
        prefix = meters[:-1]  # obtained by slicing the k-mer from the beginning to the second-to-last character
        suffix = meters[1:]  # obtained by slicing the k-mer from the second character to the end
        # check if the prefix is already in the graph. If not, we add it as a key to the graph dictionary with an empty list as its value.
        # This prepares the graph to store the suffixes associated with this prefix.
        if prefix not in graphs:
            graphs[prefix] = []
        graphs[prefix].append(suffix)  ##append the suffix to the list of neighbors for the corresponding prefix in the graph
    return graphs  # return the completed de Bruijn grap




"""
Represents the DFS algortihm, passing a graph already built
"""


def depth_first_search(graph, start, visited=None):
    if visited is None:  # check if a none was already visited
        # if true, we initialize it as an empty set.
        # This is done to ensure that the same visited set is not shared across multiple calls to the dfs() function.
        visited = set()
    # add the nodes that were visited
    visited.add(start)
    #print the start of the nodes, ends with theres no nodes that were visited
    print(start, end=" ")
    #We iterate over each neighbor of the start node.
    #We use graph.get(start, []) to retrieve the list of neighbors of the start node from the graph.
    #If the start node is not found in the graph, an empty list [] is returned.
    for neighbor in graph.get(start, []):
        #For each neighbor of the start node, we check if it has not been visited before.
        if neighbor not in visited:
            #If the neighbor has not been visited, we recursively call the depth_first_search() function with the neighbor as the new starting node.
            # This continues the depth_first_search traversal from the neighbor node.
            depth_first_search(graph, neighbor, visited)



def main():

    #ori = 'TATGGGGTG'
    #print(built_ori_from_string(ori))

    K_meter_genome = "GAGG CAGG GGGG GGGA CAGG AGGG GGAG".split()
    #K_meter_genome = "GCAAG CAGCT TGACG".split()

    graph_output = bruijn_graph(K_meter_genome)
    print('Output', graph_output)
    start_node = next(iter(graph_output.keys())) #obtain the first start node of the graph
    print('Depth First Search:')
    depth_first_search(graph_output, start_node)

if __name__ == "__main__":
    main()



