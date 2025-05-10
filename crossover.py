import utils 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

# ----------------------------------------------------------
# Crossover Operators
# ----------------------------------------------------------

"""
Order crossover (CX)

Parent1: [1, 2, 3, 4, 5, 6, 7, 8, 9]
Parent2: [9, 5, 8, 2, 1, 7, 6, 3, 4]

index xo points (inclusive): 3 - 6 

Parent1: [1, 2, 3 | 4, 5, 6, 7 | 8, 9]
Parent2: [9, 5, 8 | 2, 1, 7, 6 | 3, 4]

CHILD 1 

Fill from parent1: [1, 2, 3 | 4, 5, 6, 7 | 8, 9]

    Child1: [_, _, _, 4, 5, 6, 7, _, _]

Fill from parent2 in order: [9, 5, 8, 2, 1, 7, 6, 3, 4]
    
    9 is not present ✅

    Child1: [9, _, _, 4, 5, 6, 7, _, _] ✅
    
    5 is present ❌

    Child1: [9, 5, _, 4, 5, 6, 7, _, _] ❌ would mean duplicate for 5 
    
    8 is not present ✅

    Child1: [9, 8, _, 4, 5, 6, 7, _, _] ✅
    
    2 is not present ✅

    Child1: [9, 8, 2, 4, 5, 6, 7, _, _] ✅
    
    1 is not present and fills the next gap ✅
    
    Child1: [9, 8, 2, 4, 5, 6, 7, 1, _] ✅
    
    7, 6 are present ❌
    
    3 is not present ✅
    
    Child1: [9, 8, 2, 4, 5, 6, 7, 1, 3] ✅
    
    4 is present and child is full ❌

    
CHILD 2
Fill from parent2: [9, 5, 8 | 2, 1, 7, 6 | 3, 4]

    Child2: [_, _, _, 2, 1, 7, 6, _, _]
    
Fill from parent1 in order: [1, 2, 3, 4, 5, 6, 7, 8, 9]

    1, 2 is present ❌
    
    3, 4, 5 is not present ✅
    
    Child2: [3, 4, 5, 2, 1, 7, 6, _, _] ✅
    
    6, 7 is present ❌
    
    8, 9 is not present ✅
    
    Child2: [3, 4, 5, 2, 1, 7, 6, 8, 9] ✅
"""


repr1 = [[5, 56, 21, 38, 48, 47, 33, 44],
 [16, 52, 61, 3, 46, 23, 7, 17],
 [60, 19, 22, 35, 51, 39, 4, 28],
 [15, 32, 29, 41, 27, 10, 54, 42],
 [40, 64, 30, 25, 49, 14, 59, 20],
 [1, 55, 26, 6, 13, 31, 8, 43],
 [11, 36, 34, 58, 12, 18, 50, 63],
 [57, 45, 24, 37, 2, 53, 62, 9]]


repr2 = [[13, 6, 46, 8, 12, 11, 28, 23],
 [48, 3, 60, 15, 39, 4, 64, 14],
 [19, 32, 5, 30, 54, 49, 9, 38],
 [43, 47, 63, 53, 36, 33, 61, 58],
 [17, 25, 1, 2, 24, 10, 26, 57],
 [20, 52, 22, 62, 44, 31, 34, 29],
 [35, 45, 27, 42, 50, 7, 59, 41],
 [37, 18, 40, 51, 21, 56, 55, 16]]

def classic_order_crossover(p1, p2):
    """
    Copy a slice from a parent; 
    Fill the rest of space (and missing values) from other parent in order 
    """
    
    p1_flatten = [g for table in p1 for g in table]
    p2_flatten = [g for table in p2 for g in table]

    # start to end copy
    start, end = sorted(random.sample(range(64), 2))
    
    child1 = [None] * 64
    child2 = [None] * 64
    
    # Copy a slice from parent to respective child 
    child1[start:end+1] = p1_flatten[start:end+1]
    child2[start:end+1] = p2_flatten[start:end+1]
    
    # Fill in the remaining spots in child1 using parent2 in order
    p2_idx = 0
    for i in range(64):
        if child1[i] is None:
            while p2_flatten[p2_idx] in child1:
                p2_idx += 1
            child1[i] = p2_flatten[p2_idx]
    
    # Fill in the remaining spots in child2 using parent1 in order
    p1_idx = 0
    for i in range(64):
        if child2[i] is None:
            while p1_flatten[p1_idx] in child2:
                p1_idx += 1
            child2[i] = p1_flatten[p1_idx]
    
    # Rebuild the 8x8 table representation for both children
    child1_repr = [child1[i:i + 8] for i in range(0, 64, 8)]
    child2_repr = [child2[i:i + 8] for i in range(0, 64, 8)]
    
    return child1_repr, child2_repr

ch1, ch2 = classic_order_crossover(repr1, repr2)
# off1, off2 = classic_order_crossover(rep1, rep2)
all(all(isinstance(person, int) for person in table) for table in ch2)


"""
Cycle crossover (CX)

Parent1 = [1, 2, 3, 4, 5, 6, 7, 8]
Parent2 = [4, 1, 2, 3, 8, 7, 6, 5]

Starting at random index parent1 eg - index 0: 

    Child1: [1, _, _, _, _, _, _, _]
    Child2: [4, _, _, _, _, _, _, _]
    
Next index parent1 where value is 4 - index 3:
    
    Child1: [1, _, _, 4, _, _, _, _]
    Child2: [4, _, _, 3, _, _, _, _]

Next index parent1 where value is 3 - index 2:

    Child1: [1, _, 3, 4, _, _, _, _]
    Child2: [4, _, 2, 3, _, _, _, _]
    
Next index parent1 where value is 2 - index 1:

    Child1: [1, 2, 3, 4, _, _, _, _]
    Child2: [4, 1, 2, 3, _, _, _, _]

Next index parent1 where value is 1 - index 0:

    The start index is reached and the cycle is complete. 

Fill the rest of the child with the values from the other parent in order:

    Child1: [1, 2, 3, 4, 8, 7, 6, 5]
    Child2: [4, 1, 2, 3, 5, 6, 7, 8]
    
    
"""

def cycle_crossover(rep1, rep2):
    parent1_flat = [guest for table in rep1 for guest in table]
    parent2_flat = [guest for table in rep2 for guest in table]
    
    # get the initial index from parent1
    initial_idx = random.randint(0, len(parent1_flat) - 1)
    cycle_idx = [initial_idx]
    curr_cycle_idx = initial_idx

    while True:
        # get the indexes from parent1 to be inserted in child1 from parent1 and in child2 from parent2
        value_from_parent2 = parent2_flat[curr_cycle_idx]
        next_cycle_idx = parent1_flat.index(value_from_parent2)

        if next_cycle_idx == initial_idx:
            break

        cycle_idx.append(next_cycle_idx)
        curr_cycle_idx = next_cycle_idx

    # Step 3: Build the offspring
    offspring1_flat = []
    offspring2_flat = []

    for i in range(len(parent1_flat)):
        if i in cycle_idx:
            # Copy from parent1 for offspring1, and from parent2 for offspring2
            offspring1_flat.append(parent1_flat[i])
            offspring2_flat.append(parent2_flat[i])
        else:
            # Swap roles for other positions
            offspring1_flat.append(parent2_flat[i])
            offspring2_flat.append(parent1_flat[i])

    # Reshape back into 8x8 tables
    offspring1 = [offspring1_flat[i:i+8] for i in range(0, 64, 8)]
    offspring2 = [offspring2_flat[i:i+8] for i in range(0, 64, 8)]

    return offspring1, offspring2

# Parent1 = [1, 2, 3, 4, 5, 6, 7, 8]
# Parent2 = [4, 1, 2, 3, 8, 7, 6, 5]

# off1, off2 = cycle_crossover(Parent1, Parent2)

# print("Parent 1:", Parent1)
# print("Parent 2:", Parent2)
# print("Offspring 1:", off1)
# print("Offspring 2:", off2)

# off1, off2 = cycle_crossover(rep1, rep2)



"""
Partially mapped 

Parent1: [1, 2, 3, 4, 5, 6, 7, 8]
Parent2: [4, 1, 2, 3, 8, 7, 6, 5]

index xo points (inclusive): 3 - 5 

Parent1: [1, 2, 3, | 4, 5, 6 |, 7, 8]
Parent2: [4, 1, 2, | 3, 8, 7 |, 6, 5]

CHILD1 
Fill from parent2 the mapping points:
    
    Child1: [_, _, _, | 3, 8, 7 |, _, _]

For the rest of the points fill child1 from parent1 in order: 
Parent1: [1, 2, 3, | 4, 5, 6 |, 7, 8]
Maps: (3 - 4), (8 - 5), (7 - 6)
 
    1, 2 is not present ✅
    
    Child1: [1, 2, _, | 3, 8, 7 |, _, _] 
    
    3 is present  ❌
    Go to mapped points and get corresponding value from parent1 - 4
    
    Child1: [1, 2, 4, | 3, 8, 7 |, _, _]
    
    7 is present ❌
    Go to mapped points and get corresponding value from parent1 - 6
    
    Child1: [1, 2, 4, | 3, 8, 7 |, 6, _]
    
    8 is present ❌
    Go to mapped points and get corresponding value from parent1 - 5
    
    Child1: [1, 2, 4, | 3, 8, 7 |, 6, 5]    
    
CHILD2
Fill from parent1 the mapping points:
    
    Child2: [_, _, _, | 4, 5, 6 |, _, _]
    
For the rest of the points fill child2 from parent2 in order:
Parent2: [4, 1, 2, | 3, 8, 7 |, 6, 5]
Maps: (4 - 3), (5 - 8), (6 - 7)

    4 is present ❌
    Go to mapped points and get corresponding value from parent2 - 3
    
    Child2: [3, _, _, | 4, 5, 6 |, _, _]
    
    1, 2 is not present ✅
    
    Child2: [3, 1, 2, | 4, 5, 6 |, _, _]
    
    6 and 5 are present ❌
    Go to mapped points and get corresponding value from parent2 - 7, 8
    
    Child2: [3, 1, 2, | 4, 5, 6 |, 7, 8]
    
"""

def has_duplicates(seating):
    """
    Checks if a seating arrangement has duplicate guests.

    Args:
        seating (list[list[int]]): A list of 8 tables, each with 8 guest IDs.

    Returns:
        bool: True if there are duplicates, False otherwise.
    """
    flat = [guest for table in seating for guest in table]
    return len(flat) != len(set(flat))

def partially_mapped_crossover(rep1, rep2):
    parent1_flat = [guest for table in rep1 for guest in table]
    parent2_flat = [guest for table in rep2 for guest in table]

    start, end = sorted(random.sample(range(64), 2))

    child1 = [None] * 64
    child2 = [None] * 64

    # Copy the mapped section from other parent  
    child1[start:end+1] = parent2_flat[start:end+1]
    child2[start:end+1] = parent1_flat[start:end+1]

    # Create mappings for each child 
    mapping1 = {parent2_flat[i]: parent1_flat[i] for i in range(start, end+1)}
    mapping2 = {parent1_flat[i]: parent2_flat[i] for i in range(start, end+1)}
    
    def resolve_conflict(val, child_window, mapping):
        # Recursively resolve mapped values until no conflict
        while val in child_window:
            val = mapping[val]
        # Exists loop when the value in the mapping is not in the child window 
        return val
    
    # Fill in remaining positions
    for i in range(64):
        if i < start or i > end:
            # For child1
            val1 = parent1_flat[i]
            if val1 not in child1:
                child1[i] = val1
            else:
                child1[i] = resolve_conflict(val1, child1[start:end+1], mapping1)

            # For child2
            val2 = parent2_flat[i]
            if val2 not in child2:
                child2[i] = val2
            else:
                child2[i] = resolve_conflict(val2, child2[start:end+1], mapping2)

    # Reshape into 8x8 tables
    offspring1 = [child1[i:i+8] for i in range(0, 64, 8)]
    offspring2 = [child2[i:i+8] for i in range(0, 64, 8)]
    
    if has_duplicates(offspring1) or has_duplicates(offspring2):
        raise ValueError("Invalid offspring — duplicates or missing guests!")
    
    return offspring1, offspring2

# off1, off2 = partially_mapped_crossover(rep1, rep2)



# def m_partial_mapped_crossover(parent1_repr, parent2_repr):
#     """
#     Como tem uma componente aleatoria; nao vou usar 
#     """
#     # Flatten both parents
#     flat1 = [g for table in parent1_repr for g in table]
#     flat2 = [g for table in parent2_repr for g in table]
    
#     # Choose two crossover points
#     start, end = sorted(random.sample(range(64), 2))

#     # Empty child
#     child = [None] * 64

#     # Copy middle section from parent1
#     child[start:end+1] = flat1[start:end+1]
#     used = set(child[start:end+1])  # Track used values

#     # Try to fill left and right from parent2 (same positions)
#     for i in list(range(0, start)) + list(range(end+1, 64)):
#         val = flat2[i]
#         if val not in used:
#             child[i] = val
#             used.add(val)

#     # Fill in the remaining positions "randomly" with unused guests
#     remaining = [g for g in range(1, 65) if g not in used]
#     random.shuffle(remaining)

#     for i in range(64):
#         if child[i] is None:
#             child[i] = remaining.pop()

#     # Rebuild 8x8 tables
#     child_repr = [child[i:i + 8] for i in range(0, 64, 8)]

#     # Final validation
#     if len(set(child)) != 64:
#         raise ValueError("Invalid offspring — duplicates or missing guests!")

#     return child_repr

