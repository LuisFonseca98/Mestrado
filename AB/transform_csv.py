import csv

# Organized data
data = [
    ("Achtman 7 Gene MLST", "aroC", ""),
    ("Achtman 7 Gene MLST", "dnaN", ""),
    ("Achtman 7 Gene MLST", "hemD", ""),
    ("Achtman 7 Gene MLST", "hisD", ""),
    ("Achtman 7 Gene MLST", "purE", ""),
    ("Achtman 7 Gene MLST", "sucA", ""),
    ("Achtman 7 Gene MLST", "thrA", "")
]

# Write data to a CSV file
filename = "dataset.csv"
with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Scheme name", "Locus tag", "Description"])  # Write header
    writer.writerows(data)  # Write data
