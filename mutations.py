import utils 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

# ----------------------------------------------------------
# Mutation Operators
# ----------------------------------------------------------

rep1 = [
    [1, 2, 3, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14, 15, 16],
    [17, 18, 19, 20, 21, 22, 23, 24],
    [25, 26, 27, 28, 29, 30, 31, 32],
    [33, 34, 35, 36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45, 46, 47, 48],
    [49, 50, 51, 52, 53, 54, 55, 56],
    [57, 58, 59, 60, 61, 62, 63, 64],
]

rep2 = [
    [64, 63, 62, 61, 60, 59, 58, 57],
    [56, 55, 54, 53, 52, 51, 50, 49],
    [48, 47, 46, 45, 44, 43, 42, 41],
    [40, 39, 38, 37, 36, 35, 34, 33],
    [32, 31, 30, 29, 28, 27, 26, 25],
    [24, 23, 22, 21, 20, 19, 18, 17],
    [16, 15, 14, 13, 12, 11, 10, 9],
    [8, 7, 6, 5, 4, 3, 2, 1],
]



def swap_mutation(representation, mut_prob):
    """
    Ensures that the swap is between two different tables.
    """
    if random.random() <= mut_prob:
        new_repr = deepcopy(representation)
        # This ensures that table1_idx != table2_idx
        table1_idx, table2_idx = random.sample(range(0,7), 2)

        #These indices can be equal        
        person1_idx = random.randint(0, 7) 
        person2_idx = random.randint(0, 7)
        
        new_repr[table1_idx][person1_idx], new_repr[table2_idx][person2_idx] = (
            new_repr[table2_idx][person2_idx],
            new_repr[table1_idx][person1_idx],
        )
        return new_repr
    return representation


mut = swap_mutation(rep1, 1)
mut
len(mut)



def insert_mutation(representation, mut_prob):
    if random.random() < mut_prob:
        # Flatten the matrix into 1D list
        flat = [guest for table in representation for guest in table]

        # This picks two different indices
        idx1, idx2 = random.sample(range(64), 2)
        guest = flat.pop(idx2)

        # Adjust idx1 if idx2<idx1
        if idx2 < idx1:
            idx1 -= 1

        flat.insert(idx1 + 1, guest)


        # Reconstruct into 8x8 matrix
        new_repr = [flat[i:i + 8] for i in range(0, 64, 8)]

        return new_repr
    
    return representation # if mutation doesn't occur it returns the original arrangement

mut = insert_mutation(rep1, 1)
mut



def inversion_mutation(representation, mut_prob):
    if random.random() < mut_prob:
        # Flatten 2D seating into 1D list
        flat = [guest for table in representation for guest in table]

        # Pick two random positions to define the inversion segment
        i, j = sorted(random.sample(range(64), 2))

        # Reverse the segment
        flat[i:j+1] = flat[i:j+1][::-1]

        # Reconstruct 8 tables of 8 guests each
        new_repr = [flat[k:k + 8] for k in range(0, 64, 8)]

        return new_repr
    return representation

mut = inversion_mutation(rep1, 1)
mut




def scramble_mutation(representation, mut_prob):
    """
    Select a random size of guests to scramble (between 2 and 64) (higher probability for smaller sizes)
    
    Randomly select the indexes of the guests to scramble, shuffle the values and reassign them to the same indexes.
    """
    if random.random() < mut_prob:
        # Flatten 8x8 representation
        flat = [guest for table in representation for guest in table]
        
        # Define the weights for the scramble size (higher weights for smaller sizes)
        weights = [1 / (i + 1) for i in range(2, 64)] 
        scramble_size = random.choices(range(2, 64), weights=weights, k=1)[0]
        
        # Select random subset of indices to scramble
        indices = random.sample(range(len(flat)), scramble_size)

        # Extract and shuffle the values
        values = [flat[i] for i in indices]
        random.shuffle(values)

        # Reassign shuffled values back into their positions
        for i, idx in enumerate(indices):
            flat[idx] = values[i]

        # Rebuild into 8 tables of 8 guests each
        new_repr = [flat[i:i + 8] for i in range(0, 64, 8)]


        return new_repr

    return representation 

mut = scramble_mutation(rep1, 1)
mut

