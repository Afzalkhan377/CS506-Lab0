## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = cosine_similarity(vector1, vector2)
    
    # Compute the expected result manually
    dot_product_val = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    expected_result = dot_product_val / (norm1 * norm2)
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    points = np.array([[1, 2], [4, 5], [7, 8]])
    query_point = np.array([5, 5])
    
    result = nearest_neighbor(points, query_point)
    
    # Compute the expected result manually
    distances = np.linalg.norm(points - query_point, axis=1)
    expected_index = np.argmin(distances)
    
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
