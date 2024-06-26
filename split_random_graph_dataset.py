import os
import random
import pickle

def split_files(directory, percentage):
    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Parcourir tous les fichiers du dossier actuel
    files = [filename for filename in os.listdir(directory) if filename.endswith('.gpickle')]

    print(f"Total files before shuffling: {len(files)}")
    random.shuffle(files)

    # Calculer le nombre de fichiers pour chaque ensemble basé sur le pourcentage
    total_files = len(files)
    subset_count = int(total_files * (percentage / 100))
    train_count = int(subset_count * 0.7)
    val_count = int(subset_count * 0.15)
    test_count = subset_count - train_count - val_count

    # Répartir les fichiers dans les listes spécifiques
    subset_files = files[:subset_count]
    train_files = subset_files[:train_count]
    val_files = subset_files[train_count:train_count + val_count]
    test_files = subset_files[train_count + val_count:]

    # Écrire les listes de fichiers dans des fichiers .pkl
    with open(os.path.join(directory, 'training_list'), 'wb') as f:
        pickle.dump(train_files, f)
    with open(os.path.join(directory, 'validation_list'), 'wb') as f:
        pickle.dump(val_files, f)
    with open(os.path.join(directory, 'test_list'), 'wb') as f:
        pickle.dump(test_files, f)

    # Écrire les listes de fichiers dans des fichiers .txt
    with open(os.path.join(directory, 'training_list.txt'), 'w') as f:
        for file in train_files:
            f.write(f"{file}\n")
    with open(os.path.join(directory, 'validation_list.txt'), 'w') as f:
        for file in val_files:
            f.write(f"{file}\n")
    with open(os.path.join(directory, 'test_list.txt'), 'w') as f:
        for file in test_files:
            f.write(f"{file}\n")

    print("Files have been split and saved into .pkl and .txt files.")

# Appeler la fonction avec le dossier cible et le pourcentage du dataset
split_files('./data/random_graph', 1)