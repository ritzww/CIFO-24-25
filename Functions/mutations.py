import utils 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from itertools import combinations

# ----------------------------------------------------------
# Mutation Operators
# ----------------------------------------------------------

rep1 = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28, 29, 30, 31],
    [32, 33, 34, 35, 36, 37, 38, 39],
    [40, 41, 42, 43, 44, 45, 46, 47],
    [48, 49, 50, 51, 52, 53, 54, 55],
    [56, 57, 58, 59, 60, 61, 62, 63],
]
rep2 = [
    [63, 62, 61, 60, 59, 58, 57, 56],
    [55, 54, 53, 52, 51, 50, 49, 48],
    [47, 46, 45, 44, 43, 42, 41, 40],
    [39, 38, 37, 36, 35, 34, 33, 32],
    [31, 30, 29, 28, 27, 26, 25, 24],
    [23, 22, 21, 20, 19, 18, 17, 16],
    [15, 14, 13, 12, 11, 10, 9, 8],
    [7, 6, 5, 4, 3, 2, 1, 0],
]


def swap_mutation(representation, mut_prob):
    """
    Ensures that the swap is between two different tables.
    """
    if random.random() <= mut_prob:
        new_repr = deepcopy(representation)
        # This ensures that table1_idx != table2_idx
        table1_idx, table2_idx = random.sample(range(0,7), 2)
        
        person1_idx = random.randint(0, 7) #These indexes can be equal
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
        # Flatten the list (8x8 → 64)
        # No need for deepcopy as we are creating already a new flattened list
        flat = [guest for table in representation for guest in table]

        idx1, idx2 = random.sample(range(64), 2)

        # Remove guest at idx2 and insert it after idx1
        guest = flat.pop(idx2)
        flat.insert(idx1 + 1, guest)

        # Reconstruct into 8 tables (list of lists)
        new_repr = [flat[i:i + 8] for i in range(0, 64, 8)]


        return new_repr
    
    return representation

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
        weights = [1 / (i) for i in range(2, 64)] 
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



def scramble_mutation_optimized(representation, mut_prob, k=2):
    """
    Select a random size of guests to scramble (between 2 and 64) (higher probability for smaller sizes)
    
    Randomly select the indexes of the guests to scramble, shuffle the values and reassign them to the same indexes.
    """
    if random.random() < mut_prob:
        # Flatten 8x8 representation
        flat = [guest for table in representation for guest in table]
        
        # Define the weights for the scramble size (higher weights for smaller sizes)
        weights = [1 / (i**k) for i in range(2, 64)] 
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



"""
The disruption is expectationally higher for the inversion mutation and scramble mutation, vs the swap mutation, which is constant (always swap two guests between two different tables).
The disruption for the scramble mutation can be null if the scrambled guests are in the same table."""


def structure_disruption(p, c):
    def get_same_table_pairs(tables):
        pairs = set()
        for table in tables:
            for g1, g2 in combinations(sorted(table), 2):
                pairs.add((g1, g2))
        return pairs

    p_pairs = get_same_table_pairs(p)
    c_pairs = get_same_table_pairs(c)
    preserved = p_pairs & c_pairs
    return 1 - (len(preserved) / len(p_pairs)) if p_pairs else 0.0


disruptions_swap = []
disruptions_scramble = []
disruptions_inversion = []

runs = 1000

for _ in range(runs):
    parent1 = list(range(1, 65))
    random.shuffle(parent1)
    parent1 = [parent1[i:i + 8] for i in range(0, 64, 8)]
    

    c1 = swap_mutation(parent1, 1)
    c2 = scramble_mutation(parent1, 1)
    c3 = inversion_mutation(parent1, 1)
    
    dist_c1_p1 = structure_disruption(parent1, c1)
    dist_c2_p1 = structure_disruption(parent1, c2)
    dist_c3_p1 = structure_disruption(parent1, c3)
    
    disruptions_swap.append(dist_c1_p1)
    disruptions_scramble.append(dist_c2_p1)
    disruptions_inversion.append(dist_c3_p1)
    


# # Plotting side-by-side boxplots
# plt.figure(figsize=(10, 6))
# plt.boxplot([disruptions_swap, disruptions_inversion, disruptions_scramble], labels=["Swap", "Inversion", "Scramble"])
# plt.ylim(0, 1)
# plt.title("Structure Disruption by Mutation Operator")
# plt.ylabel("Disruption (0 = none, 1 = full)")
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()


"""

$$
w_i = \frac{1}{i^k}
$$

Where:

* $w_i$ is the weight for scramble size $i$
* $k$ is the exponent that controls the decay rate (e.g., bigger k gives stronger preference to smaller scramble sizes)

For the GA we will use a k=1 (high scramble size), and for SA we will test for different k values (1, 2, 3). 

"""

disruptions_scramble_1 = []
disruptions_scramble_2 = []
disruptions_scramble_3 = []

runs = 1000

for _ in range(runs):
    parent1 = list(range(1, 65))
    random.shuffle(parent1)
    parent1 = [parent1[i:i + 8] for i in range(0, 64, 8)]
    

    c1 = scramble_mutation_optimized(parent1, 1, k=1)
    c2 = scramble_mutation_optimized(parent1, 1, k=2)
    c3 = scramble_mutation_optimized(parent1, 1, k=3)
    
    dist_c1_p1 = structure_disruption(parent1, c1)
    dist_c2_p1 = structure_disruption(parent1, c2)
    dist_c3_p1 = structure_disruption(parent1, c3)
    
    disruptions_scramble_1.append(dist_c1_p1)
    disruptions_scramble_2.append(dist_c2_p1)
    disruptions_scramble_3.append(dist_c3_p1)
    


# plt.figure(figsize=(10, 6))
# plt.boxplot([disruptions_scramble_1, disruptions_scramble_2, disruptions_scramble_3], labels=["Scramble (k=1)", "Scramble (k=2)", "Scramble (k=3)"])
# plt.ylim(0, 1)
# plt.title("Structure Disruption by Scramble k Value")
# plt.ylabel("Disruption (0 = none, 1 = full)")
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()
