import numpy as np
from typing import List

def calculate_sentence_length_percentile(sentence_lengths: List[int], percentile_value: float) -> int:
    """
    Calculates the specified percentile length from a list of sentence lengths.

    Args:
        sentence_lengths (List[int]): A list of integers representing the lengths of sentences.
        percentile_value (float): The desired percentile to calculate (0-100).

    Returns:
        int: The length corresponding to the specified percentile.
    """
    if not sentence_lengths:
        raise ValueError("The list of sentence lengths cannot be empty.")  # Ensure the input list is not empty

    # Calculate the specified percentile length
    percentile_length = int(np.percentile(sentence_lengths, percentile_value))
    return percentile_length  # Return the calculated length

