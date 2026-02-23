import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import random
from lineworking import (
    fitness, initialize_population, tournament_selection,
    blend_crossover, smooth_mutation
)

# Track waypoints
x = np.array([0, 20, 40, 60, 80, 100, 120, 140])
y = np.array([0, 15, 30, 10, -10, -5, -5, -5])

# Interpolate centerline
X_ = np.linspace(x.min(), x.max(), 200)
cubic_model = interp1d(x, y, kind="cubic")
Y_ = cubic_model(X_)

# Calculate track normals
dx = np.gradient(X_)
dy = np.gradient(Y_)
norm = np.sqrt(dx**2 + dy**2)
dx_norm = dx / norm
dy_norm = dy / norm

# Normal vectors (pointing left)
nx = -dy_norm
ny = dx_norm

# Track parameters
TRACK_WIDTH = 10.0
HALF_WIDTH = TRACK_WIDTH / 2
LOWER_BOUND = -HALF_WIDTH
UPPER_BOUND = HALF_WIDTH

# GA Parameters
POP_SIZE = 200
N_STATIONS = len(X_)
NUM_GENERATIONS = 600
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.20
TOURNAMENT_SIZE = 7
ELITE_COUNT = 10

# Initialize
population = initialize_population(POP_SIZE, N_STATIONS, LOWER_BOUND, UPPER_BOUND)
best_individual = None
best_fitness = 0.0

# Main loop
for gen in range(NUM_GENERATIONS):

    # Evaluate fitness
    fitness_values = []
    for ind in population:
        fit = fitness(ind, X_, Y_, nx, ny, track_width=HALF_WIDTH)
        fitness_values.append(fit)

    fitness_values = np.array(fitness_values)

    # Track best
    gen_best_idx = np.argmax(fitness_values)
    gen_best_fitness = fitness_values[gen_best_idx]
    gen_best_individual = population[gen_best_idx]

    # Update global best
    if gen_best_fitness > best_fitness:
        best_fitness = gen_best_fitness
        best_individual = gen_best_individual.copy()

    # Selection
    selected = tournament_selection(population, fitness_values, TOURNAMENT_SIZE)

    # Create next generation
    next_generation = []

    # Elitism
    elite_indices = np.argsort(fitness_values)[-ELITE_COUNT:]
    for idx in elite_indices:
        next_generation.append(population[idx].copy())

    # Crossover
    while len(next_generation) < POP_SIZE:
        parent1 = random.choice(selected)
        parent2 = random.choice(selected)

        if random.random() < CROSSOVER_RATE:
            child1, child2 = blend_crossover(parent1, parent2)
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        next_generation.append(child1)
        if len(next_generation) < POP_SIZE:
            next_generation.append(child2)

    next_generation = next_generation[:POP_SIZE]

    # Mutation
    sigma = 3.0 * (1 - gen / NUM_GENERATIONS) + 0.5

    mutated = []
    for i, ind in enumerate(next_generation):
        if i < ELITE_COUNT:
            mutated.append(ind)
        else:
            mut_ind = smooth_mutation(ind, MUTATION_RATE, LOWER_BOUND, UPPER_BOUND, sigma)
            mutated.append(mut_ind)

    population = mutated

# Convert to coordinates
racing_x = X_ + best_individual * nx
racing_y = Y_ + best_individual * ny
inner_x = X_ - nx * HALF_WIDTH
inner_y = Y_ - ny * HALF_WIDTH
outer_x = X_ + nx * HALF_WIDTH
outer_y = Y_ + ny * HALF_WIDTH

# Plot
plt.figure(figsize=(14, 10))
plt.plot(outer_x, outer_y, 'k-', linewidth=3, label='Track boundary')
plt.plot(inner_x, inner_y, 'k-', linewidth=3)
plt.plot(X_, Y_, '--', color='blue', linewidth=2, alpha=0.5, label='Centerline')
plt.plot(racing_x, racing_y, 'r-', linewidth=3, label='Racing line')
plt.scatter(racing_x[0], racing_y[0], c='green', s=300, zorder=15)
plt.title('Optimized Racing Line', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
