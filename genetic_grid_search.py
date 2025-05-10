
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from crossover import *
from problems_solutions import *
from selection_algos import *
from mutations import *
from algorithms import *
from copy import deepcopy
from functools import partial
import os
from itertools import product
import json
import math
from utils import *

scores = pd.read_csv('Wedding/seating_data(in).csv', index_col=0)
scores_array = scores.to_numpy()
scores_array[0][0]

# Parameter grid
pop_sizes = [10, 50, 250]
mutation_ops = [swap_mutation, inversion_mutation, scramble_mutation] # insert_mutation
crossover_ops = [classic_order_crossover, partially_mapped_crossover] # cycle_crossover 
selection_names = ["tournament", "rank"] # fitness
max_gens = [500]
elitisms = [True] # False with very low results
xo_probs = [0.8, 0.7] #0.9 
mut_probs = [0.2, 0.3] # 0.1 
tourn_ks = [5, 7] # 3 
# tourn_replacement = [True, False] 
rank_ls = [1] # 1e-2, 0.1 # 1e-2 similar to linear 




"""
We will load the result from the CSV file to store the tested configurations and save those in a set of tuples. 
"""
results_file = "ga_results.csv"


# Load or initialize the results dataframe
if os.path.exists(results_file):
    df_existing = pd.read_csv(results_file)
    tested_configs = set(config_to_key(row) for _, row in df_existing.iterrows())
else:
    df_existing = pd.DataFrame()
    tested_configs = set()


"""
In the grid search loop, we will iterate over all combinations of the parameters. 
Each combination of parameters will be ran 30 times to obtain a large enough sample size to compare the results. 
The results will be saved in a CSV file, so that we can load it later and append new results.
If combination of parameters has already been tested, we will skip it, allowing efficient exploration of the hyperparameter space.
"""

# Grid search loop
for pop_size, mutation_fn, crossover_fn, selection_name, max_gen, elitism, xo_prob, mut_prob in product(
    pop_sizes, mutation_ops, crossover_ops, selection_names, max_gens, elitisms, xo_probs, mut_probs
):

    # Skip if incompatible selection and missing param
    if selection_name == "tournament":
        selection_configs = tourn_ks
    elif selection_name == "rank":
        selection_configs = rank_ls
    else:
        selection_configs = [None]  # fitness has no extra param

    for sel_config in selection_configs:
        # Create a tuple key for the current configurations 
        current_key = (
            pop_size,
            mutation_fn.__name__,
            crossover_fn.__name__,
            selection_name,
            sel_config,
            elitism,
            max_gen,
            round(xo_prob, 4),
            round(mut_prob, 4)
        )
        
        # If the current configuration has already been tested (in tested_configs), skip it
        if current_key in tested_configs:
            print(f"Skipping already tested config: {current_key}")
            continue
        
        # Set selection algorithm hyperparameters
        if selection_name == "tournament":
            selection_algorithm = partial(tournament_selection, k=sel_config)
        elif selection_name == "fitness":
            selection_algorithm = fitness_proportionate_selection
        else:  # rank
            selection_algorithm = partial(rank_selection, function="exponential", l=sel_config)

        # Repeat GA run 30 times for each configuration
        fitness_scores = []
        conv_gens = []
        conv_times = []
        
        print(f"Running GA with config: {current_key}")
        for _ in range(30):
            population = [
                Wedding_GA_Solution(
                    scores=scores_array,
                    mutation_function=mutation_fn,
                    crossover_function=crossover_fn
                ) for _ in range(pop_size)
            ]

            best_sol, _, conv_gen, conv_time = genetic_algorithm(
                initial_population=deepcopy(population),
                max_gen=max_gen,
                selection_algorithm=selection_algorithm,
                maximization=True,
                xo_prob=xo_prob,
                mut_prob=mut_prob,
                elitism=elitism,
                verbose=False
            )

            fitness_scores.append(best_sol.fitness())
            conv_gens.append(conv_gen)
            conv_times.append(round(conv_time, 2))
            
        avg_fitness = round(np.mean(fitness_scores), 2)
        std_fitness = round(np.std(fitness_scores),2)
        
        avg_conv_gen = round(np.mean(conv_gens),2)
        avg_conv_time = round(np.mean(conv_times),2)
        
        print(f"Avg fitness: {round(avg_fitness, 1)}, Std: {round(std_fitness,1)} in {avg_conv_gen} generations, {avg_conv_time} seconds")
        
        df_new = pd.DataFrame([{
            "pop_size": pop_size,
            "mutation": mutation_fn.__name__,
            "crossover": crossover_fn.__name__,
            "selection": selection_name,
            "selection_param": sel_config,
            "elitism": elitism,
            "max_gen": max_gen,
            "xo_prob": round(xo_prob, 4),
            "mut_prob": round(mut_prob, 4),
            "avg_fitness": avg_fitness,
            "std_fitness": std_fitness,
            "avg_conv_gen": avg_conv_gen,
            "avg_conv_time": avg_conv_time,
            "fitness_scores": json.dumps([float(x) for x in fitness_scores]),
        }])
        
        # Append to existing CSV (or write new one)
        if os.path.exists(results_file) and not df_new.empty:
            # mode = "a" = append
            df_new.to_csv(results_file, mode='a', header=False, index=False) 
        elif not df_new.empty:
            df_new.to_csv(results_file, index=False)
        
        tested_configs.add(current_key)
        





df_ = pd.read_csv(results_file)

df = df_[df_["pop_size"] == 10]
  
df[df["crossover"] == "cycle_crossover"]["avg_fitness"].mean()
df[df["crossover"] == "classic_order_crossover"]["avg_fitness"].mean()
df[df["crossover"] == "partially_mapped_crossover"]["avg_fitness"].mean()     
# errorbar

df[(df["selection"] == "rank") & (df["selection_param"] == 1)]["avg_fitness"].mean()
df[(df["selection"] == "rank") & (df["selection_param"] == 0.1)]["avg_fitness"].mean()


df[(df["selection"] == "fitness")]["avg_fitness"].mean()

df[(df["selection"] == "tournament") & (df["selection_param"] == 5)]["avg_fitness"].mean()
df[(df["selection"] == "tournament") & (df["selection_param"] == 7)]["avg_fitness"].mean()

df[df["elitism"] == True]["avg_fitness"].mean()
df[df["elitism"] == False]["avg_fitness"].mean()


df[df["mut_prob"] == 0.1]["avg_fitness"].mean()
df[df["mut_prob"] == 0.2]["avg_fitness"].mean()

df[df["xo_prob"] == 0.8]["avg_fitness"].mean()
df[df["xo_prob"] == 0.9]["avg_fitness"].mean()