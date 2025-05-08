import utils 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from problems_solutions import *
import math 

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
        
        



def rank_selection(population: list[Solution], function, maximization: bool, l=0.2):
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






def tournament_selection(population: list[Solution], k, maximization: bool):
    if k >= len(population):
        raise ValueError("Tournament size k must be smaller than the population size")

    tournament_group = random.choices(population, k=k)

    if maximization:
        winner = max(tournament_group, key=lambda ind: ind.fitness())
    else:
        winner = min(tournament_group, key=lambda ind: ind.fitness())

    return deepcopy(winner)

