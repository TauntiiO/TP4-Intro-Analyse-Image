import cv2
import numpy as np
from collections import deque
import time  # Pour mesurer le temps d'exécution

def cc_label(binary_matrix):
    """
    Étiquetage en composantes connexes avec 4-adjacence.
    
    :param binary_matrix: Matrice binaire (numpy array) où les pixels blancs (255) sont les objets.
    :return: Matrice labélisée où chaque composante connexe a un label unique.
    """
    rows, cols = binary_matrix.shape
    labeled_matrix = np.zeros((rows, cols), dtype=np.int32) 
    visited = np.zeros((rows, cols), dtype=np.uint8)       
    label = 1  # Compteur de labels

    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(rows):
        for x in range(cols):
            if binary_matrix[y, x] == 255 and not visited[y, x]:
                queue = deque()
                queue.append((y, x))
                visited[y, x] = 1
                labeled_matrix[y, x] = label

                while queue:
                    current_y, current_x = queue.popleft()

                    for dy, dx in neighbours:
                        next_y, next_x = current_y + dy, current_x + dx

                        if 0 <= next_y < rows and 0 <= next_x < cols:
                            if binary_matrix[next_y, next_x] == 255 and not visited[next_y, next_x]:
                                visited[next_y, next_x] = 1
                                labeled_matrix[next_y, next_x] = label
                                queue.append((next_y, next_x))

                label += 1

    return labeled_matrix


def cc_area_filter(labeled_matrix, threshold):
    """
    Filtre les composantes connexes dont la taille est inférieure au seuil donné.
    
    :param labeled_matrix: Matrice labélisée (numpy array) où chaque composante connexe a un label unique.
    :param threshold: Seuil de taille (nombre de pixels) pour conserver une composante connexe.
    :return: Matrice binaire filtrée où les composantes trop petites sont supprimées.
    """
    # Calculer la taille de chaque composante connexe
    unique_labels, counts = np.unique(labeled_matrix, return_counts=True)

    unique_labels = unique_labels[1:]  # Ignorer le label 0 (background)
    counts = counts[1:]  # Ignorer le label 0 (background)
    
    # Créer un masque pour les labels à conserver
    mask = np.isin(labeled_matrix, unique_labels[counts >= threshold])
    
    # Appliquer le masque pour créer une image binaire filtrée
    filtered_matrix = np.where(mask, 255, 0).astype(np.uint8)
    
    return filtered_matrix


def display_matrix(matrix, scale_factor=20):
    """
    Affiche une matrice en agrandissant l'image pour une meilleure visualisation.
    
    :param matrix: Matrice à afficher.
    :param scale_factor: Facteur d'agrandissement de l'image.
    """
    if matrix.dtype != np.uint8:
        matrix = (matrix * 255 / matrix.max()).astype(np.uint8)

    resized_image = cv2.resize(matrix, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Matrix", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class UnionFind:
    """
    Structure de données Union-Find pour gérer les équivalences entre labels.
    """
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        # Trouver la racine de l'ensemble contenant x
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        elif self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Compression de chemin
        return self.parent[x]

    def union(self, x, y):
        # Fusionner les ensembles contenant x et y
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1


def cc_two_pass_label(binary_matrix):
    """
    Labélisation en composantes connexes en 2 passes.
    
    :param binary_matrix: Matrice binaire (numpy array) où les pixels blancs (255) sont les objets.
    :return: Matrice labélisée où chaque composante connexe a un label unique.
    """
    rows, cols = binary_matrix.shape
    labeled_matrix = np.zeros((rows, cols), dtype=np.int32)  # Matrice de sortie pour les labels
    uf = UnionFind()  # Structure Union-Find pour gérer les équivalences
    next_label = 1  # Compteur de labels

    # Première passe : attribution des labels et gestion des équivalences
    for y in range(rows):
        for x in range(cols):
            if binary_matrix[y, x] == 255:  # Ignorer les pixels de fond (0)
                # Voisins déjà labélisés (haut et gauche)
                neighbors = []
                if y > 0 and labeled_matrix[y - 1, x] != 0:
                    neighbors.append(labeled_matrix[y - 1, x])
                if x > 0 and labeled_matrix[y, x - 1] != 0:
                    neighbors.append(labeled_matrix[y, x - 1])

                if not neighbors:
                    # Nouvelle composante connexe
                    labeled_matrix[y, x] = next_label
                    next_label += 1
                else:
                    # Attribuer le label minimum parmi les voisins
                    min_label = min(neighbors)
                    labeled_matrix[y, x] = min_label

                    # Fusionner les labels des voisins
                    for neighbor in neighbors:
                        uf.union(min_label, neighbor)

    # Deuxième passe : résolution des équivalences
    for y in range(rows):
        for x in range(cols):
            if labeled_matrix[y, x] != 0:
                labeled_matrix[y, x] = uf.find(labeled_matrix[y, x])

    return labeled_matrix


# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de matrice binaire (0 et 255)
    binary_matrix = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 255, 255, 0, 255, 0, 0],
        [0, 255, 0, 0, 255, 0, 0],
        [0, 0, 0, 0, 255, 255, 0],
        [0, 0, 255, 255, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ], dtype=np.uint8)

    # Mesurer le temps d'exécution de cc_label
    start_time = time.time()
    labeled_matrix_cc = cc_label(binary_matrix)
    cc_label_time = time.time() - start_time

    print("Matrice labélisée (cc_label) :")
    print(labeled_matrix_cc)
    display_matrix(labeled_matrix_cc, scale_factor=40)

    # Mesurer le temps d'exécution de cc_two_pass_label
    start_time = time.time()
    labeled_matrix_two_pass = cc_two_pass_label(binary_matrix)
    cc_two_pass_time = time.time() - start_time

    print("Matrice labélisée (cc_two_pass_label) :")
    print(labeled_matrix_two_pass)
    display_matrix(labeled_matrix_two_pass, scale_factor=40)

    # Afficher les temps d'exécution
    print(f"Temps d'exécution de cc_label : {cc_label_time:.6f} secondes")
    print(f"Temps d'exécution de cc_two_pass_label : {cc_two_pass_time:.6f} secondes")