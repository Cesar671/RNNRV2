import os
import numpy as np
from preprocessing.PreProcesamiento import extraer_mfcc, rellenar_caracteristicas

print("creando dataset...")

root_dir = 'samples'
features_list = []
labels_list = []
max_timesteps = 0
mfcc_dim = 20

for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)

    if os.path.isdir(folder_path):  # Verificar si es una carpeta
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                features = extraer_mfcc(file_path)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(folder_name)  # La etiqueta es el nombre de la carpeta
                    if len(features) > max_timesteps:
                        max_timesteps = len(features)

# Convertir las listas de características y etiquetas a un DataFrame de pandas
padded_features = rellenar_caracteristicas(features_list, max_timesteps, mfcc_dim)

features_array = np.array(padded_features, dtype=object)
labels_array = np.array(labels_list)

output_npz = 'caracteristicas_audios.npz'
np.savez(output_npz, features=features_array, labels=labels_array)
print("dataset creado...")
