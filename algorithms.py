import random
from copy import deepcopy
from typing import Callable
from problems_solutions import *
from selection_algos import * 
import time



def get_best_ind(population: list[Solution], maximization: bool):
    fitness_list = [ind.fitness() for ind in population]
    if maximization:
        return population[fitness_list.index(max(fitness_list))]
    else:
        return population[fitness_list.index(min(fitness_list))]




def genetic_algorithm(
    initial_population: list[Solution],
    max_gen: int,
    selection_algorithm: Callable,
    maximization: bool = False,
    xo_prob: float = 0.9,
    mut_prob: float = 0.2,
    elitism: bool = True,
    verbose: bool = False,
):
    """
    Executes a genetic algorithm to optimize a population of solutions.

    Args:
        initial_population (list[Solution]): The starting population of solutions.
        max_gen (int): The maximum number of generations to evolve.
        selection_algorithm (Callable): Function used for selecting individuals.
        maximization (bool, optional): If True, maximizes the fitness function; otherwise, minimizes. Defaults to False.
        xo_prob (float, optional): Probability of applying crossover. Defaults to 0.9.
        mut_prob (float, optional): Probability of applying mutation. Defaults to 0.2.
        elitism (bool, optional): If True, carries the best individual to the next generation. Defaults to True.
        verbose (bool, optional): If True, prints detailed logs for debugging. Defaults to False.

    Returns:
        Solution: The best solution found on the last population after evolving for max_gen generations.
    """
    # 1. Initialize a population with N individuals
    population = initial_population
    
    best_fitness_history = []
    best_solution = None

    # For convergence tracking
    best_fitness = None
    convergence_gen = 0
    convergence_time = 0
    start_time = time.time()
    
    # 2. Repeat until termination condition
    for gen in range(1, max_gen + 1):

        # 2.1. Create an empty population P'
        new_population = []

        # 2.2. If using elitism, insert best individual from P into P'
        if elitism:
            new_population.append(deepcopy(get_best_ind(population, maximization)))
        
        # 2.3. Repeat until P' contains N individuals
        while len(new_population) < len(population):
            # 2.3.1. Choose 2 individuals from P using a selection algorithm
            first_ind = selection_algorithm(population=population, maximization=maximization)
            second_ind = selection_algorithm(population=population, maximization=maximization)

            # 2.3.2. Choose an operator between crossover and replication
            # 2.3.3. Apply the operator to generate the offspring
            if random.random() < xo_prob:
                offspring1, offspring2 = first_ind.crossover(second_ind)
            else:
                offspring1, offspring2 = deepcopy(first_ind), deepcopy(second_ind)
            
            # 2.3.4. Apply mutation to the offspring
            first_new_ind = offspring1.mutation(mut_prob)
            # 2.3.5. Insert the mutated individuals into P'
            new_population.append(first_new_ind)

            
            if len(new_population) < len(population):
                second_new_ind = offspring2.mutation(mut_prob)
                new_population.append(second_new_ind)
        
        # 2.4. Replace P with P'
        population = new_population
        
        current_best = get_best_ind(population, maximization)
        current_fitness = current_best.fitness()
        best_fitness_history.append(current_fitness)

        # Update best solution and convergence info
        if (best_fitness is None) or (maximization and current_fitness > best_fitness) or (not maximization and current_fitness < best_fitness):
            best_fitness = current_fitness
            best_solution = deepcopy(current_best)
            convergence_gen = gen
            convergence_time = time.time() - start_time

        if verbose:
            print(f'Best individual in gen {gen} with fitness: {get_best_ind(population, maximization).fitness()}')

    # 3. Return the best individual in P
    return best_solution, best_fitness_history, convergence_gen, convergence_time
