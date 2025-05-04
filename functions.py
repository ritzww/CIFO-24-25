import random
import math
import numpy as np
import pandas as pd
from copy import deepcopy
from utils import Solution 

# --------------------------------------------------------------------------------------------
# 2. Wedding Solution Representation
# --------------------------------------------------------------------------------------------

class WeddingSolution(Solution):
    def __init__(
        self,
        scores: pd.DataFrame | np.ndarray = scores,
        repr: list = None,
    ):
        
        if repr:
            repr = self._validate_repr(repr)
                
        self.scores = self._convert_scores(scores)
        
        super().__init__(repr=repr)
    
    
    def __repr__(self):
        repr_str = ""
        for idx, table in enumerate(self.repr, start=1):
            repr_str += f"\nTable {idx}: {table}"
        return repr_str
    
    
    def _validate_repr(self, repr):
        if len(repr) != 8:
            raise ValueError("Representation must be a list of 8 tables")
        if not all(isinstance(table, list) for table in repr):
            raise TypeError("Each table must be a list")
        if not all(all(isinstance(person, int) for person in table) for table in repr):
            raise TypeError("Each person in the table must be an integer")
        if not all(len(table) == 8 for table in repr):
            raise ValueError("Each table must have 8 people")
        return repr
    
    
    def _convert_scores(self, scores):
        if isinstance(scores, pd.DataFrame):
            return scores.to_numpy()
        elif isinstance(scores, np.ndarray):
            return scores
        else:
            raise TypeError("Scores must be a DataFrame or a numpy array")
    
    
    def random_initial_representation(self):
        representation = []
        all_people = list(range(1, 65))
        
        for i in range(8):
            table = random.sample(all_people, 8)
            all_people = [person for person in all_people if person not in table]
            representation.append(table)
        return representation
     
     
    def fitness(self):
        total_score = 0
        for table in self.repr: 
            for i in range(len(table)):
                for j in range(i+1, len(table)):
                    # only considers the upper triangle of the scores
                    total_score += self.scores[table[i]-1][table[j]-1]
        return total_score
    
# --------------------------------------------------------------------------------------------
# 3. Wedding Genetic Algorithm
# --------------------------------------------------------------------------------------------

class WeddingGeneticAlgorithm(WeddingSolution):
    def __init__(
        self, 
        mutation_function,
        crossover_function,
        repr=None, 
        scores: pd.DataFrame | np.ndarray = scores,
    ):
        super().__init__(
            repr=repr,
            scores=scores,
        )
        
        self.mutation_function = mutation_function
        self.crossover_function = crossover_function
        
    def mutation(self, mut_prob):
        new_repr = self.mutation_function(self.repr, mut_prob)
        
        return WeddingGeneticSolution(
            repr=new_repr,
            mutation_function=self.mutation_function,
            crossover_function=self.crossover_function,
            scores=self.scores
        )
        
    def crossover(self, other_solution):
        new_repr = self.crossover_function(self.repr, other_solution.repr)
        
        return (
            WeddingGeneticSolution(
                repr=new_repr,
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
                scores=self.scores
            ),
            WeddingGeneticSolution(
                repr=other_solution.repr,
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
                scores=self.scores
            )
        )
    
# --------------------------------------------------------------------------------------------
# 4.1 Fitness Proportionate Selection
# --------------------------------------------------------------------------------------------

def fitness_proportionate_selection(population: list[Solution], maximization:bool):
    fitness_values = [ind.fitness() for ind in population]

    min_fitness = min(fitness_values)
    # Check if there are non-positive fitness values
    if min_fitness <=0:
        shift = abs(min_fitness)+1
        fitness_values = [f + shift for f in fitness_values]

    if not maximization:
        fitness_values = [1/(f) for f in fitness_values] # no need to worry about zero division, as we have +1
    
    total_fitness = sum(fitness_values)
    
    # Generate random number between 0 and total
    random_nr = random.uniform(0, total_fitness)
    # For each individual, check if random number is inside the individual's "box"
    box_boundary = 0
    for ind_idx, ind in enumerate(population):
        box_boundary += fitness_values[ind_idx]
        if random_nr <= box_boundary:
            return deepcopy(ind)
        

# --------------------------------------------------------------------------------------------
# 4.2 Rank Selection
# --------------------------------------------------------------------------------------------

def rank_selection(population: list[Solution], function, maximization: bool,l=0.2):
    """
    - function: linear or exponential
    - True(1) = maximization
    - l - lambda for the exponential function, no need to add for linear
    """
    n = len(population)
    
    ranking = [i for i in range(1, n + 1)]
    
    if maximization:
        sorted_population = sorted(population, key=lambda ind: ind.fitness())
    else:
        sorted_population = sorted(population, key=lambda ind: ind.fitness(), reverse=True)

    probabilities = []

    if function == 'linear':
        denominator = sum(ranking)
        for rank in ranking:
            probabilities.append(rank / denominator)

    elif function == 'exponential':

        denominator = sum(math.exp(-l * (n - rank)) for rank in ranking)

        for rank in ranking:
            prob = math.exp(-l * (n - rank)) / denominator
            probabilities.append(prob)

    # Generate random number between 0 and 1, since it's probabilities
    random_nr = random.uniform(0, 1)

    box_boundary = 0
    for ind_idx, ind in enumerate(sorted_population):
        box_boundary += probabilities[ind_idx]
        if random_nr <= box_boundary:
            return deepcopy(ind)

# --------------------------------------------------------------------------------------------
# 4.3 Tournament selection
# --------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------
# 4.4 Another one
# --------------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------------
# 5.1 Classic order crossover
# --------------------------------------------------------------------------------------------

def classic_order_crossover(p1, p2):
    # Flatten the parents (make them 1D lists of guests)
    p1_flatten = [g for table in p1 for g in table]
    p2_flatten = [g for table in p2 for g in table]

    
    start, end = sorted(random.sample(range(64), 2))
    
    # Initialize the children (with None as placeholders)
    child1 = [None] * 64
    child2 = [None] * 64
    
    # Copy a slice from parent 1 to child 1 (from start to end)
    child1[start:end+1] = p1_flatten[start:end+1]
    
    # Copy a slice from parent 2 to child 2 (from start to end)
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


# --------------------------------------------------------------------------------------------
# 5.2 Partial mapped crossover
# --------------------------------------------------------------------------------------------

def m_partial_mapped_crossover(parent1_repr, parent2_repr):
    """
    Como tem uma componente aleatoria; nao vou usar 
    """
    # Flatten both parents
    flat1 = [g for table in parent1_repr for g in table]
    flat2 = [g for table in parent2_repr for g in table]
    
    # Choose two crossover points
    start, end = sorted(random.sample(range(64), 2))

    # Empty child
    child = [None] * 64

    # Copy middle section from parent1
    child[start:end+1] = flat1[start:end+1]
    used = set(child[start:end+1])  # Track used values

    # Try to fill left and right from parent2 (same positions)
    for i in list(range(0, start)) + list(range(end+1, 64)):
        val = flat2[i]
        if val not in used:
            child[i] = val
            used.add(val)

    # Fill in the remaining positions "randomly" with unused guests
    remaining = [g for g in range(1, 65) if g not in used]
    random.shuffle(remaining)

    for i in range(64):
        if child[i] is None:
            child[i] = remaining.pop()

    # Rebuild 8x8 tables
    child_repr = [child[i:i + 8] for i in range(0, 64, 8)]

    # Final validation
    if len(set(child)) != 64:
        raise ValueError("Invalid offspring — duplicates or missing guests!")

    return child_repr


# --------------------------------------------------------------------------------------------
# 6.1 Swap mutation
# --------------------------------------------------------------------------------------------

def swap_mutation(representation, mut_prob):
    if random.random() <= mut_prob:
        table1_idx = random.randint(0, 7)
        table2_idx = random.randint(0, 7)
        
        person1_idx = random.randint(0, 7)
        person2_idx = random.randint(0, 7)
        
        representation[table1_idx][person1_idx], representation[table2_idx][person2_idx] = (
            representation[table2_idx][person2_idx],
            representation[table1_idx][person1_idx],
        )
    return representation


# --------------------------------------------------------------------------------------------
# 6.2 Insert mudation
# --------------------------------------------------------------------------------------------

def insert_mutation(representation, mut_prob):
    if random.random() < mut_prob:
        # Flatten the list (8x8 → 64)
        flat = [guest for table in representation for guest in table]

        # Pick two unique indices
        while True:
            idx1, idx2 = sorted(random.sample(range(64), 2))
            if idx1 != idx2:
                break 

        # Remove guest at idx2 and insert it after idx1
        guest = flat.pop(idx2)
        flat.insert(idx1 + 1, guest)

        # Reconstruct into 8 tables
        new_repr = [flat[i:i + 8] for i in range(0, 64, 8)]

        # Sanity check
        if len(set(flat)) != 64:
            raise ValueError("Duplicate or missing guests after insert mutation.")

        return new_repr
    
    return representation

# --------------------------------------------------------------------------------------------
# 6.3 Inversion mutation
# --------------------------------------------------------------------------------------------

def inversion_mutation(representation, mut_prob):
    if random.random() < mut_prob:
        # Flatten the 2D seating list into a 1D list
        flat = [guest for table in representation for guest in table]

        # Choose two positions to define the inversion segment
        start, end = sorted(random.sample(range(len(flat)), 2))

        # Invert the sublist between the two indices
        flat[start:end+1] = flat[start:end+1][::-1]

        # Reconstruct 8 tables of 8 guests each
        new_repr = [flat[i:i + 8] for i in range(0, 64, 8)]

        # Sanity check
        if len(set(flat)) != 64:
            raise ValueError("Duplicate or missing guests after inversion mutation.")

        return new_repr

    return representation


# --------------------------------------------------------------------------------------------
# 6.4 Scramble mutation
# --------------------------------------------------------------------------------------------

def scramble_mutation(representation, mut_prob):
    if random.random() < mut_prob:
        # Flatten 8x8 representation
        flat = [guest for table in representation for guest in table]
        
        # Define the weights for the scramble size (higher weights for smaller sizes)
        weights = [1 / (i**3) for i in range(2, 64)] 
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

        # Sanity check
        if len(set(flat)) != 64:
            raise ValueError("Duplicate or missing guests after scramble mutation.")

        return new_repr

    return representation 