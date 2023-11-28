import math
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import copy
# Randomly generate a population of chromosomes
def generate_population(size):
    population = []
    for _ in range(size):
        chromosome = random.sample(range(convenience_num),AED_num)
        chromosome.sort()
        population.append(chromosome)
    return population

# Calculate the fitness score of each chromosome
def fitness_calc(chromosome):
    wheight = 0
    for gen in range(ohca_num):
        distance_list = [gps_distance[gen][i] for i in chromosome]
        # Count coverable OHCA cases
        #wheight+= 1 if min(distance_list) <=distance_threshold else 0
        wheight+= 1/(min(distance_list)**2) if min(distance_list) <=distance_threshold else 0
    return wheight

# Roulette wheel selection method
def roulette_wheel_selection(population, fitness):
    total_fitness = sum(fitness)
    # Selection probability of each candidate site
    probabilities = [f / total_fitness for f in fitness]

    cumulative_probabilities = []
    cumulative_probability = 0
    for probability in probabilities:
        cumulative_probability += probability
        cumulative_probabilities.append(cumulative_probability)

    selected_individuals = []
    for _ in range(2):
        r = random.random()
        for i, probability in enumerate(cumulative_probabilities):
            if r <= probability:
                selected_individuals.append(population[i])
                break
    return selected_individuals

def count_greater(lst, start, end):
    count = 0
    for element in lst[start:end]:
        if element > -1:
            count += 1
    return count

def crossover(pop,crossover_rate):
    parent1, parent2 = pop
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    if random.random() < crossover_rate:
        parent1_alignment = []
        parent2_alignment = [] 
        for i in range(convenience_num):
            if i in parent1:
                parent1_alignment.append(i)
            else:
                parent1_alignment.append(-1)
            if i in parent2:
                parent2_alignment.append(i)
            else:
                parent2_alignment.append(-1)
        for i in range(convenience_num):
            if not parent1_alignment[i] == parent2_alignment[i]:
                break
        start_pos = i
        end_pos_list = []
        for j in range(start_pos+1,convenience_num):
            count_gene_1 = count_greater(parent1_alignment,start_pos,j)
            count_gene_2 = count_greater(parent2_alignment,start_pos,j)
            if count_gene_1 == count_gene_2 and count_gene_1>0:
                end_pos_list.append(j)
        if len(end_pos_list)==0:
            return child1,child2
        end_pos = random.choice(end_pos_list)
        start_pos_list = []
        for j in range(start_pos,end_pos):
            count_gene_1 = count_greater(parent1_alignment,j,end_pos)
            count_gene_2 = count_greater(parent2_alignment,j,end_pos)
            if count_gene_1 == count_gene_2 and count_gene_1>0:
                start_pos_list.append(j)
        start_pos = random.choice(start_pos_list)
        parent1_alignment[start_pos:end_pos],parent2_alignment[start_pos:end_pos] = parent2_alignment[start_pos:end_pos],parent1_alignment[start_pos:end_pos]
        child1 = [g for g in parent1_alignment if g>-1]
        child2 = [g for g in parent2_alignment if g>-1]
    return child1,child2

def mutate(child,mutation_rate):
    if random.random() < mutation_rate:
        gene_list = list(set([i for i in range(convenience_num)]) - set(child))
        gene = random.choice(gene_list)
        child[random.randint(0,len(child)-1)] = gene
    return child

# Calculate distance between two latitude-longitude points
def haversine(lat1, lng1, lat2, lng2):
    # convert decimal degrees to radians
    lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
    # haversine formula
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
    c = 2 * math.asin(math.sqrt(a))
    meter = 6367000 * c
    return meter

if __name__ == "__main__":
    # Read data file
    convenience_gps = pd.read_csv("./convenience_shop.csv").values
    ohca_gps = pd.read_csv("./OHCA_location.csv").values
    convenience_num = convenience_gps.shape[0]
    ohca_num = ohca_gps.shape[0]
    gps_distance = [[0 for i in range(convenience_num)] for j in range(ohca_num)]
    for i in range(ohca_num):
        for j in range(convenience_num):
            gps_distance[i][j] = haversine(ohca_gps[i][0], ohca_gps[i][1], convenience_gps[j][0], convenience_gps[j][1])
    size = 400
    maxgens = 400
    crossover_rate = 0.8
    mutate_rate = 0.01
    distance_threshold = 100
    AED_num = 50
    population = generate_population(size)
    fitness_best_list = []
    fitness_best = -np.inf
    for generation in range(maxgens):
        fitness_list = []
        for chromosome in population:
            fitness_list.append(fitness_calc(chromosome))
        fitness_best_list.append(max(fitness_list))
        if max(fitness_list) > fitness_best:
            fitness_best = max(fitness_list)
            best_pop = population[fitness_list.index(max(fitness_list))]
        print(f"Generation {generation + 1} The highest weight：{fitness_best}")
        # Roulette wheel selection method
        
        offspring = [best_pop]
        while len(offspring) < size:
            parents = roulette_wheel_selection(population, fitness_list)
            while parents[0] == parents[1]:
                parents = roulette_wheel_selection(population, fitness_list)
            # Crossover
            child1,child2 = crossover(parents,crossover_rate)
            # Mutation
            child1 = mutate(child1,mutate_rate)
            child2 = mutate(child2,mutate_rate)
            offspring.append(child1)
            offspring.append(child2)
        population = copy.deepcopy(offspring)
    print(f"The best fitness score：{fitness_best}")
    print(f"Optimal soultions：{best_pop}")
    plt.figure()
    plt.title(f'Fitness')
    plt.plot(fitness_best_list)
    plt.grid(True)
    plt.show()
