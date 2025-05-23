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


repr1 = [n for n in range(1, 65)]
random.shuffle(repr1)
repr1 = [repr1[i:i + 8] for i in range(0, 64, 8)]
# print(repr1)

repr2 = [n for n in range(1, 65)]
random.shuffle(repr2)
repr2 = [repr2[i:i + 8] for i in range(0, 64, 8)]
# print(repr2)
# def classic_order_crossover(p1, p2):
#     """
#     Copy a slice from a parent; 
#     Fill the rest of space (and missing values) from other parent in order 
#     """
    
#     p1_flatten = [g for table in p1 for g in table]
#     p2_flatten = [g for table in p2 for g in table]

#     # start to end copy
#     start, end = sorted(random.sample(range(64), 2))
    
#     child1 = [None] * 64
#     child2 = [None] * 64
    
#     # Copy a slice from parent to respective child 
#     child1[start:end+1] = p1_flatten[start:end+1]
#     child2[start:end+1] = p2_flatten[start:end+1]
    
#     # Fill in the remaining spots in child1 using parent2 in order
#     p2_idx = 0
#     for i in range(64):
#         if child1[i] is None:
#             while p2_flatten[p2_idx] in child1:
#                 p2_idx += 1
#             child1[i] = p2_flatten[p2_idx]
    
#     # Fill in the remaining spots in child2 using parent1 in order
#     p1_idx = 0
#     for i in range(64):
#         if child2[i] is None:
#             while p1_flatten[p1_idx] in child2:
#                 p1_idx += 1
#             child2[i] = p1_flatten[p1_idx]
    
#     # Rebuild the 8x8 table representation for both children
#     child1_repr = [child1[i:i + 8] for i in range(0, 64, 8)]
#     child2_repr = [child2[i:i + 8] for i in range(0, 64, 8)]
    
#     return child1_repr, child2_repr

# ch1, ch2 = classic_order_crossover(repr1, repr2)
# # off1, off2 = classic_order_crossover(rep1, rep2)
# all(all(isinstance(person, int) for person in table) for table in ch2)

from itertools import combinations

def classic_order_crossover(p1, p2, verbose=False):
    """
    Performs classic Order Crossover (OX) with shuffled guest indices.
    Input: list-of-lists representation (8 tables of 8 guests).
    Output: two valid children (same structure).
    """
    # Step 1: Flatten parents
    p1_flat = [g for table in p1 for g in table]
    p2_flat = [g for table in p2 for g in table]

    # Step 2: Shuffle guest indices
    shuffled_indices = list(range(64))
    random.shuffle(shuffled_indices)

    p1_shuffled = [p1_flat[i] for i in shuffled_indices]
    p2_shuffled = [p2_flat[i] for i in shuffled_indices]

    # Step 3: Choose crossover segment
    start, end = sorted(random.sample(range(64), 2))

    child1_shuffled = [None] * 64
    child2_shuffled = [None] * 64

    # Copy the crossover segment
    child1_shuffled[start:end+1] = p1_shuffled[start:end+1]
    child2_shuffled[start:end+1] = p2_shuffled[start:end+1]

    # Fill in the remaining spots from the other parent
    def fill_ox(target, donor):
        donor_idx = 0
        for i in range(64):
            if target[i] is None:
                while donor[donor_idx] in target:
                    donor_idx += 1
                target[i] = donor[donor_idx]

    fill_ox(child1_shuffled, p2_shuffled)
    fill_ox(child2_shuffled, p1_shuffled)

    # Step 4: Unshuffle — map back to original guest positions
    child1 = [None] * 64
    child2 = [None] * 64
    for i, idx in enumerate(shuffled_indices):
        child1[idx] = child1_shuffled[i]
        child2[idx] = child2_shuffled[i]

    # Step 5: Rebuild 8x8 table format
    child1_repr = [child1[i:i + 8] for i in range(0, 64, 8)]
    child2_repr = [child2[i:i + 8] for i in range(0, 64, 8)]
    
    if verbose:
        def get_pairs(rep):
            pairs = set()
            for table in rep:
                for a, b in combinations(sorted(table), 2):
                    pairs.add((a, b))
            return pairs

        parent_pairs = get_pairs(p1)
        child1_pairs = get_pairs(child1_repr)
        child2_pairs = get_pairs(child2_repr)

        same1 = len(parent_pairs & child1_pairs)
        same2 = len(parent_pairs & child2_pairs)
        total = len(parent_pairs)

        print(f"Child 1: {same1}/{total} guest pairs ({same1/total:.1%}) stayed together")
        print(f"Child 2: {same2}/{total} guest pairs ({same2/total:.1%}) stayed together")
        
    return child1_repr, child2_repr

def group_preserving_order_crossover(p1, p2):

    def build_child(preserve_from, fill_from):
        # Shuffle both parent table orders
        preserve_tables = deepcopy(preserve_from)
        fill_tables = deepcopy(fill_from)
        random.shuffle(preserve_tables)
        random.shuffle(fill_tables)

        # Randomly choose how many tables to preserve
        num_preserve = random.randint(1, 7)
        preserve_indices = sorted(random.sample(range(8), num_preserve))
        preserved = [preserve_tables[i] for i in preserve_indices]
        preserved_guests = set(g for table in preserved for g in table)

        # Flatten the fill-from parent and remove preserved guests
        fill_order = [g for table in fill_tables for g in table if g not in preserved_guests]

        # Fill remaining tables
        remaining_tables = [fill_order[i*8:(i+1)*8] for i in range(8 - num_preserve)]

        # Combine preserved and new tables
        child = preserved + remaining_tables
        random.shuffle(child)

        # Validation
        flat = [g for t in child for g in t]
        assert len(flat) == 64 and len(set(flat)) == 64, "Invalid child: duplicates or missing guests"
        return child


    child1 = build_child(p1, p2)
    child2 = build_child(p2, p1)
    return child1, child2

# p1 = [i for i in range(1, 65)]
# random.shuffle(p1)
# p1 = np.reshape(p1, (8, 8)).tolist()
# p2 = [i for i in range(1, 65)]
# random.shuffle(p2)
# p2 = np.reshape(p2, (8, 8)).tolist()

# c1, c2 = group_preserving_order_crossover(p1, p2)
# print("Parent 1:")
# print(p1)
# print("Parent 2:")
# print(p2)
# print("Child 1:")
# print(c1)
# print("Child 2:")
# print(c2)


# ch1, ch2 = classic_order_crossover(repr1, repr2, verbose=True)
# print("parent1")
# print(repr1)
# print("parent2")
# print(repr2)
# print("child1")
# print(ch1)
# print("child2")
# print(ch2)

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


# def partially_mapped_crossover(rep1, rep2):
    
#     parent1_flat = [guest for table in rep1 for guest in table]
#     parent2_flat = [guest for table in rep2 for guest in table]

#     start, end = sorted(random.sample(range(64), 2))

#     child1 = [None] * 64
#     child2 = [None] * 64

#     # Copy the mapped section from other parent  
#     child1[start:end+1] = parent2_flat[start:end+1]
#     child2[start:end+1] = parent1_flat[start:end+1]

#     # Create mappings for each child 
#     mapping1 = {parent2_flat[i]: parent1_flat[i] for i in range(start, end+1)}
#     mapping2 = {parent1_flat[i]: parent2_flat[i] for i in range(start, end+1)}
    
#     def resolve_conflict(val, child_window, mapping):
#         # Recursively resolve mapped values until no conflict
#         while val in child_window:
#             val = mapping[val]
#         # Exists loop when the value in the mapping is not in the child window 
#         return val
    
#     # Fill in remaining positions
#     for i in range(64):
#         if i < start or i > end:
#             # For child1
#             val1 = parent1_flat[i]
#             if val1 not in child1:
#                 child1[i] = val1
#             else:
#                 child1[i] = resolve_conflict(val1, child1[start:end+1], mapping1)

#             # For child2
#             val2 = parent2_flat[i]
#             if val2 not in child2:
#                 child2[i] = val2
#             else:
#                 child2[i] = resolve_conflict(val2, child2[start:end+1], mapping2)

#     # Reshape into 8x8 tables
#     offspring1 = [child1[i:i+8] for i in range(0, 64, 8)]
#     offspring2 = [child2[i:i+8] for i in range(0, 64, 8)]
    

#     return offspring1, offspring2




def partially_mapped_crossover(rep1, rep2):
    parent1_flat = [guest for table in rep1 for guest in table]
    parent2_flat = [guest for table in rep2 for guest in table]

    # Step 1: Shuffle indices
    shuffled_indices = list(range(64))
    random.shuffle(shuffled_indices)

    p1_shuffled = [parent1_flat[i] for i in shuffled_indices]
    p2_shuffled = [parent2_flat[i] for i in shuffled_indices]

    # Step 2: Choose crossover segment
    start, end = sorted(random.sample(range(64), 2))
    child1_shuffled = [None] * 64
    child2_shuffled = [None] * 64

    # Step 3: Copy mapped section from other parent
    child1_shuffled[start:end+1] = p2_shuffled[start:end+1]
    child2_shuffled[start:end+1] = p1_shuffled[start:end+1]

    # Step 4: Create mappings
    mapping1 = {p2_shuffled[i]: p1_shuffled[i] for i in range(start, end+1)}
    mapping2 = {p1_shuffled[i]: p2_shuffled[i] for i in range(start, end+1)}

    def resolve_conflict(val, mapped_section, mapping):
        while val in mapped_section:
            val = mapping[val]
        return val

    # Step 5: Fill remaining positions
    for i in range(64):
        if i < start or i > end:
            # For child1
            val1 = p1_shuffled[i]
            if val1 not in child1_shuffled:
                child1_shuffled[i] = val1
            else:
                child1_shuffled[i] = resolve_conflict(val1, child1_shuffled[start:end+1], mapping1)

            # For child2
            val2 = p2_shuffled[i]
            if val2 not in child2_shuffled:
                child2_shuffled[i] = val2
            else:
                child2_shuffled[i] = resolve_conflict(val2, child2_shuffled[start:end+1], mapping2)

    # Step 6: Unshuffle to original guest order
    child1 = [None] * 64
    child2 = [None] * 64
    for i, idx in enumerate(shuffled_indices):
        child1[idx] = child1_shuffled[i]
        child2[idx] = child2_shuffled[i]

    # Step 7: Rebuild into 8x8 tables
    offspring1 = [child1[i:i+8] for i in range(0, 64, 8)]
    offspring2 = [child2[i:i+8] for i in range(0, 64, 8)]

    return offspring1, offspring2



def pmx_table_block_crossover(p1, p2):

    # Step 0: Shuffle table order to break position bias
    p1_ = deepcopy(p1)
    p2_ = deepcopy(p2)
    random.shuffle(p1_)
    random.shuffle(p2_)
    
    # Flatten both parents
    p1_flat = [g for table in p1_ for g in table]
    p2_flat = [g for table in p2_ for g in table]
    
    # Randomly choose number of contiguous tables for PMX mapping
    num_tables = random.randint(1, 7)  # can't be 8 or you'll map all tables
    
    # Step 1: Randomly choose the starting table index
    start_table = random.randint(0, 8 - num_tables)  # ensures room for n tables
    
    # Step 2: Calculate flat index range for contiguous tables
    start = start_table * 8
    end = (start_table + num_tables) * 8 - 1  # inclusive

    # Step 2: Initialize children
    child1 = [None] * 64
    child2 = [None] * 64

    # Step 3: Copy PMX segment from opposite parent
    child1[start:end+1] = p2_flat[start:end+1]
    child2[start:end+1] = p1_flat[start:end+1]

    # Step 4: Build mapping dictionaries
    mapping1 = {p2_flat[i]: p1_flat[i] for i in range(start, end+1)}
    mapping2 = {p1_flat[i]: p2_flat[i] for i in range(start, end+1)}

    # If a value in the mapping already exists in the child 
    def resolve_conflict(val, mapping, segment_vals):
        while val in segment_vals:
            val = mapping[val]
        return val

    # Step 5: Fill the rest of child1
    for i in range(64):
        if i < start or i > end:
            val1 = p1_flat[i]
            val2 = p2_flat[i]

            # For child1
            if val1 not in child1:
                child1[i] = val1
            else:
                child1[i] = resolve_conflict(val1, mapping1, child1[start:end+1])

            # For child2
            if val2 not in child2:
                child2[i] = val2
            else:
                child2[i] = resolve_conflict(val2, mapping2, child2[start:end+1])

    # Step 6: Rebuild into 8x8 tables
    child1_repr = [child1[i:i + 8] for i in range(0, 64, 8)]
    child2_repr = [child2[i:i + 8] for i in range(0, 64, 8)]

    return child1_repr, child2_repr

# off1, off2 = partially_mapped_crossover(rep1, rep2)

# Randomly choose number of contiguous tables for PMX mapping
num_tables = random.randint(1, 7)  # can't be 8 or you'll map all tables

# Step 1: Randomly choose the starting table index
start_table = random.randint(0, 8 - num_tables)  # ensures room for n tables

# Step 2: Calculate flat index range for contiguous tables
start = start_table * 8
end = (start_table + num_tables) * 8 - 1  # inclusive


# print(f"PMX on tables {start_table} to {start_table + num_tables - 1}")
# print(f"Flat indices {start} to {end}")

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

