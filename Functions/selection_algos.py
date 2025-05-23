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





    



def rank_selection_optimized(population: list, function, maximization: bool, l=0.2, fitness_list=None):
    """
    - function: 'linear' or 'exponential'
    - maximization: True for maximization problems
    - l: lambda for exponential rank-based selection
    - fitness_list: list of fitness values in the same order as population
    """

    n = len(population)
    ranking = list(range(1, n + 1))

    # Use provided fitness_list or compute it
    if fitness_list is None:
        fitness_list = [ind.fitness() for ind in population]

    # Sort population and fitness values accordingly
    if maximization:
        paired = sorted(zip(population, fitness_list), key=lambda x: x[1])
    else:
        paired = sorted(zip(population, fitness_list), key=lambda x: x[1], reverse=True)

    sorted_population, sorted_fitness = zip(*paired)

    # Compute rank-based selection probabilities
    probabilities = []

    if function == 'linear':
        denominator = sum(ranking)
        probabilities = [rank / denominator for rank in ranking]

    elif function == 'exponential':
        denominator = sum(math.exp(-l * (n - rank)) for rank in ranking)
        probabilities = [math.exp(-l * (n - rank)) / denominator for rank in ranking]

    # Roulette wheel selection
    random_nr = random.uniform(0, 1)
    box_boundary = 0
    for ind_idx, ind in enumerate(sorted_population):
        box_boundary += probabilities[ind_idx]
        if random_nr <= box_boundary:
            return deepcopy(ind)




def tournament_selection(population: list[Solution], k, maximization: bool, fitness_list:list):
    if k >= len(population):
        raise ValueError("Tournament size k must be smaller than the population size")

    tournament_group = random.choices(population, k=k)

    if maximization:
        winner = max(tournament_group, key=lambda ind: ind.fitness())
    else:
        winner = min(tournament_group, key=lambda ind: ind.fitness())

    return deepcopy(winner)



def tournament_selection_optimized(population: list[Solution], k, maximization: bool, fitness_list: list):
    if k >= len(population):
        raise ValueError("Tournament size k must be smaller than the population size")

    # Sample k indices with replacement
    indices = random.choices(range(len(population)), k=k)

    # Pair individuals directly with fitness
    tournament = [(population[i], fitness_list[i]) for i in indices]

    # Select the best based on fitness
    if maximization:
        best = max(tournament, key=lambda x: x[1])
    else:
        best = min(tournament, key=lambda x: x[1])
    return deepcopy(best[0])