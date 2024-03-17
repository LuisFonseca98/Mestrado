from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)

    def remove_edge(self, u, v):
        self.graph[u].remove(v)
        self.graph[v].remove(u)

    def euler_util(self, u, circuit):
        for v in self.graph[u]:
            if (u, v) not in circuit:
                circuit[(u, v)] = True
                self.remove_edge(u, v)
                self.euler_util(v, circuit)

    def find_euler_path(self):
        circuit = {}
        self.euler_util(list(self.graph.keys())[0], circuit)
        return list(circuit.keys())

# Function to construct the graph from the input strings
def construct_graph(input_str):
    g = Graph()
    for i in range(len(input_str) - 1):
        node1 = input_str[i][:3]
        node2 = input_str[i][1:]
        g.add_edge(node1, node2)
    return g

# Example input
input_str = "GAGG CAGG GGGG GGGA CAGG AGGG GGAG"

# Constructing the graph
g = construct_graph(input_str.split())

# Finding Euler Path
euler_path = g.find_euler_path()

# Printing the Euler Path
print("Euler Path:", euler_path)
