from sklearn.cluster import KMeans
from itertools import permutations
import numpy as np

def preprocess_dataset_from_file(filename):
    """
    Preprocess the dataset from a text file to represent strains as binary vectors.
    """
    # Initialize empty lists to store strains and genes
    strains = []
    genes = []

    # Read data from the text file
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()  # Remove leading/trailing whitespaces and newline characters
            parts = line.split()  # Split the line into parts based on whitespace

            # Check if the line contains exactly two parts (strain and gene)
            if len(parts) != 2:
                raise ValueError(f"Invalid format in line {line_number}: {line}")

            strain, gene = parts
            strains.append(strain)
            genes.append(gene)

    # Convert strains and genes into sets to get unique values
    unique_strains = sorted(set(strains))
    unique_genes = sorted(set(genes))

    # Create a dictionary to map genes to indices
    gene_to_index = {gene: index for index, gene in enumerate(unique_genes)}

    # Initialize an empty list to store binary vectors
    binary_vectors = []

    # Iterate over strains to create binary vectors
    for strain in unique_strains:
        binary_vector = [0] * len(unique_genes)
        for gene in genes:
            if gene in gene_to_index and strain == strains[genes.index(gene)]:
                gene_index = gene_to_index[gene]
                binary_vector[gene_index] = 1
        binary_vectors.append(binary_vector)

    return np.array(binary_vectors)


def robinson_foulds(cluster1, cluster2):
    """
    Compute Robinson-Foulds metric between two clusterings.
    """
    n = len(cluster1)
    if n != len(cluster2):
        raise ValueError("Both clusterings must have the same number of elements")

    # Convert clusterings into sets of sets
    c1 = [set(x) for x in cluster1]
    c2 = [set(x) for x in cluster2]

    # Compute the Robinson-Foulds distance
    rf_dist = 0
    for perm in permutations(range(n)):
        d = 0
        for i in range(n):
            j = perm[i]  # j is the index of the corresponding cluster in the permutation
            d = max(d, len((set(cluster1[i]) ^ set(cluster1[j])) | (set(range(n)) - set(cluster1[i])) - (
                        (set(cluster2[i]) ^ set(cluster2[j])) | (set(range(n)) - set(cluster2[i])))))
        rf_dist = max(rf_dist, d)

    return rf_dist

# Sample usage
filename = "dataset.csv"  # Replace with the actual filename
binary_vectors = preprocess_dataset_from_file(filename)

# Apply clustering (example using KMeans)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(binary_vectors)

# Group strains based on clusters
clustered_indices = [[] for _ in range(2)]
for index, cluster_id in enumerate(clusters):
    clustered_indices[cluster_id].append(index)

# Compute Robinson-Foulds distance between clusters
rf_distance = robinson_foulds(clustered_indices, [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11]])
print("Robinson-Foulds distance:", rf_distance)
