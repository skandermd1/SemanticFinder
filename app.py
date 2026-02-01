import math

import numpy as np
import scipy
import torch
from sentence_transformers import SentenceTransformer
documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(documents)
embeddings.shape

def euclidean_distance_fn(vector1, vector2):
    squared_sum = sum((x - y) ** 2 for x, y in zip(vector1, vector2))
    return math.sqrt(squared_sum)
euclidean_distance_fn(embeddings[0], embeddings[1])
l2_dist_manual = np.zeros([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        l2_dist_manual[i,j] = euclidean_distance_fn(embeddings[i], embeddings[j])
l2_dist_manual_improved = np.zeros([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        if j > i: # Calculate the upper triangle only
            l2_dist_manual_improved[i,j] = euclidean_distance_fn(embeddings[i], embeddings[j])
        elif i > j: # Copy the uper triangle to the lower triangle
            l2_dist_manual_improved[i,j] = l2_dist_manual[j,i]
def dot_product_fn(vector1, vector2):
    return sum(x * y for x, y in zip(vector1, vector2))
dot_product_fn(embeddings[0], embeddings[1])
dot_product_manual = np.empty([4,4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        dot_product_manual[i,j] = dot_product_fn(embeddings[i], embeddings[j])

normalized_embeddings_torch = torch.nn.functional.normalize(
    torch.from_numpy(embeddings)
).numpy()
cosine_sim_manual = np.empty([4,4])
for i in range(normalized_embeddings_torch.shape[0]):
    for j in range(normalized_embeddings_torch.shape[0]):
        cosine_sim_manual[i,j] = dot_product_fn(
            normalized_embeddings_torch[i],
            normalized_embeddings_torch[j]
        )
        # First, embed the query:
query_embedding = model.encode(
    ["do warm summer months can make bugs?"]
)

# Second, normalize the query embedding:
normalized_query_embedding = torch.nn.functional.normalize(
    torch.from_numpy(query_embedding)
).numpy()

# Third, calculate the cosine similarity between the documents and the query by using the dot product:
cosine_similarity_q3 = normalized_embeddings_torch @ normalized_query_embedding.T

# Fourth, find the position of the vector with the highest cosine similarity:
highest_cossim_position = cosine_similarity_q3.argmax()

# Fifth, find the document in that position in the `documents` array:
documents[highest_cossim_position]

# As you can see, the query retrieved the document `Bugs introduced by the intern had to be squashed by the lead developer.` which is what we would expect.
print(documents[highest_cossim_position])