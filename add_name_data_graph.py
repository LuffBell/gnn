import os
import pickle
import networkx as nx

def add_filename_to_graph(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.graphml'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as f:
                graph = nx.read_graphml(f)
                # graph = pickle.load(f)
            graph.graph['filename'] = filename
            with open(file_path, 'wb') as f:
                pickle.dump(graph, f)
            print(f"Modified: {filename}")

# Utilisation
directory = './data/rome'
add_filename_to_graph(directory)