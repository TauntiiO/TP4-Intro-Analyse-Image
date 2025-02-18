import cv2
import numpy as np
from collections import deque
import time 

def cc_label(binary_matrix):
    rows, cols = binary_matrix.shape
    labeled_matrix = np.zeros((rows, cols), dtype=np.int32) 
    visited = np.zeros((rows, cols), dtype=np.uint8)       
    label = 1 

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
    unique_labels, counts = np.unique(labeled_matrix, return_counts=True)

    unique_labels = unique_labels[1:]
    counts = counts[1:] 
    
    mask = np.isin(labeled_matrix, unique_labels[counts >= threshold])
    
    filtered_matrix = np.where(mask, 255, 0).astype(np.uint8)
    
    return filtered_matrix


def display_matrix(matrix, scale_factor=20):
    if matrix.dtype != np.uint8:
        matrix = (matrix * 255 / matrix.max()).astype(np.uint8)

    resized_image = cv2.resize(matrix, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Matrix", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        elif self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x]) 
        return self.parent[x]

    def union(self, x, y):
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
    rows, cols = binary_matrix.shape
    labeled_matrix = np.zeros((rows, cols), dtype=np.int32)  
    uf = UnionFind() 
    next_label = 1 

    for y in range(rows):
        for x in range(cols):
            if binary_matrix[y, x] == 255: 
                neighbors = []
                if y > 0 and labeled_matrix[y - 1, x] != 0:
                    neighbors.append(labeled_matrix[y - 1, x])
                if x > 0 and labeled_matrix[y, x - 1] != 0:
                    neighbors.append(labeled_matrix[y, x - 1])

                if not neighbors:
                    labeled_matrix[y, x] = next_label
                    next_label += 1
                else:
                    min_label = min(neighbors)
                    labeled_matrix[y, x] = min_label

                    for neighbor in neighbors:
                        uf.union(min_label, neighbor)

    for y in range(rows):
        for x in range(cols):
            if labeled_matrix[y, x] != 0:
                labeled_matrix[y, x] = uf.find(labeled_matrix[y, x])

    return labeled_matrix


binary_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 255, 255, 0, 255, 0, 0],
    [0, 255, 0, 0, 255, 0, 0],
    [0, 0, 0, 0, 255, 255, 0],
    [0, 0, 255, 255, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
], dtype=np.uint8)

start_time = time.time()
labeled_matrix_cc = cc_label(binary_matrix)
cc_label_time = time.time() - start_time

print("Matrice labélisée (cc_label) :")
print(labeled_matrix_cc)
display_matrix(labeled_matrix_cc, scale_factor=40)

start_time = time.time()
labeled_matrix_two_pass = cc_two_pass_label(binary_matrix)
cc_two_pass_time = time.time() - start_time

print("Matrice labélisée (cc_two_pass_label) :")
print(labeled_matrix_two_pass)
display_matrix(labeled_matrix_two_pass, scale_factor=40)

print(f"Temps d'exécution de cc_label : {cc_label_time:.6f} secondes")
print(f"Temps d'exécution de cc_two_pass_label : {cc_two_pass_time:.6f} secondes")

threshold = 3
filtered_matrix = cc_area_filter(labeled_matrix_cc, threshold)
print(f"Matrice filtrée (cc_area_filter) avec seuil {threshold} :")
print(filtered_matrix)
display_matrix(filtered_matrix, scale_factor=40)