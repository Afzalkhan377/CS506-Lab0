import numpy as np

def dot_product(v1, v2):
    '''
    v1 and v2 are vectors of the same shape.
    Return the scalar dot product of the two vectors.
    '''
    return np.dot(v1, v2)

def cosine_similarity(v1, v2):
    '''
    v1 and v2 are vectors of the same shape.
    Return the cosine similarity between the two vectors.
    '''
    # Cosine similarity formula: (v1 dot v2) / (||v1|| * ||v2||)
    dot_prod = dot_product(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    return dot_prod / (norm_v1 * norm_v2)

def nearest_neighbor(target_vector, vectors):
    '''
    target_vector is a vector of shape d.
    vectors is a matrix of shape N x d.
    Return the row index of the vector in vectors that is closest to
    target_vector in terms of cosine similarity.
    '''
    best_index = -1
    best_similarity = -1  # Since cosine similarity ranges from -1 to 1
    
    for i, vec in enumerate(vectors):
        similarity = cosine_similarity(target_vector, vec)
        if similarity > best_similarity:
            best_similarity = similarity
            best_index = i
    
    return best_index
