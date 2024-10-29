import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Scores for the students
students = ['Student1', 'Student2', 'Student3', 'Student4']
scores = [75, 80, 65, 95]

# Reshape the scores into a 2D array (required for linkage function)
scores_2d = np.array(scores).reshape(-1, 1)

# Perform hierarchical clustering using single linkage
linked = linkage(scores_2d, method='single', metric='euclidean')

# Plot the dendrogram
plt.figure(figsize=(8, 6))
dendrogram(linked,
           labels=students,
           distance_sort='ascending',
           show_leaf_counts=True)

plt.title('Dendrogram for Student Scores')
plt.xlabel('Students')
plt.ylabel('Euclidean Distance')
plt.show()
