import os
import random
import pickle

def split_files(directory):
    # Vérifier si le dossier de destination existe, sinon le créer
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Parcourir tous les fichiers du dossier actuel
    files = [filename for filename in os.listdir(directory) if filename.endswith('.gpickle')]

    print(files)
    random.shuffle(files)

    # Calculer le nombre de fichiers pour chaque ensemble
    total_files = len(files)
    train_count = int(total_files * 0.7)
    val_count = int(total_files * 0.15)
    test_count = total_files - train_count - val_count

    # Répartir les fichiers dans les listes spécifiques
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]

    # Écrire les listes de fichiers dans des fichiers .pkl
    with open(os.path.join(directory, 'training_list.pkl'), 'wb') as f:
        pickle.dump(train_files, f)
    with open(os.path.join(directory, 'validation_list.pkl'), 'wb') as f:
        pickle.dump(val_files, f)
    with open(os.path.join(directory, 'test_list.pkl'), 'wb') as f:
        pickle.dump(test_files, f)

    print("Files have been split and saved into .pkl files.")

# Appeler la fonction avec le dossier cible
split_files('./data/random_graph')