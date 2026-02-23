import numpy as np
import random
from scipy.ndimage import gaussian_filter1d


def fitness(alpha, X_, Y_, nx, ny, track_width=5.0):
    alpha = np.array(alpha)

    if np.any(np.abs(alpha) > track_width):
        return 0.0

    # Construct racing line
    racing_x = X_ + alpha * nx
    racing_y = Y_ + alpha * ny

    # Calculate curvature
    dx = np.gradient(racing_x)
    dy = np.gradient(racing_y)
    ds = np.sqrt(dx ** 2 + dy ** 2) + 1e-10
    dx_ds = dx / ds
    dy_ds = dy / ds
    ddx_ds = np.gradient(dx_ds)
    ddy_ds = np.gradient(dy_ds)
    curvature = np.abs(dx_ds * ddy_ds - dy_ds * ddx_ds)

    # Segment distances
    distances = np.hypot(np.diff(racing_x), np.diff(racing_y))

    # Maximum cornering speed
    mu = 1.5
    g = 9.81
    max_speed_limit = 80.0

    v_max = np.where(
        curvature > 1e-6,
        np.sqrt(mu * g / curvature),
        max_speed_limit
    )
    v_max = np.minimum(v_max, max_speed_limit)

    a_accel = 5.0
    a_brake = 12.0
    n = len(v_max)

    # Forward pass
    v_forward = np.zeros(n)
    v_forward[0] = v_max[0]
    for i in range(1, n):
        v_accel = np.sqrt(v_forward[i - 1] ** 2 + 2 * a_accel * distances[i - 1])
        v_forward[i] = min(v_max[i], v_accel)

    # Backward pass
    v_backward = np.zeros(n)
    v_backward[-1] = v_max[-1]
    for i in range(n - 2, -1, -1):
        v_brake = np.sqrt(v_backward[i + 1] ** 2 + 2 * a_brake * distances[i])
        v_backward[i] = min(v_max[i], v_brake)

    v = np.minimum(v_forward, v_backward)

    # Calculate lap time
    v_avg = (v[:-1] + v[1:]) / 2.0
    v_avg = np.maximum(v_avg, 1.0)
    segment_times = distances / v_avg
    lap_time = np.sum(segment_times)

    # Small penalty for extreme jerkiness
    alpha_changes = np.abs(np.diff(alpha))
    extreme_penalty = np.sum(alpha_changes > 3.0) * 10.0

    # Bonus for using track width
    avg_usage = np.mean(np.abs(alpha)) / track_width
    usage_bonus = -1.0 * avg_usage

    total_cost = lap_time + extreme_penalty + usage_bonus

    return 1.0 / (total_cost + 1e-8)


def initialize_population(pop_size, n_stations, lower, upper):
    population = []

    # Aggressive random lines (40%)
    n_aggressive = int(pop_size * 0.4)
    for _ in range(n_aggressive):
        alpha = np.random.uniform(lower, upper, n_stations)
        alpha = gaussian_filter1d(alpha, sigma=5)
        population.append(alpha)

    # Sinusoidal patterns (20%)
    n_sine = int(pop_size * 0.2)
    for _ in range(n_sine):
        freq = np.random.uniform(0.5, 3.0)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(lower, upper)
        x = np.linspace(0, freq * 2 * np.pi, n_stations)
        alpha = amplitude * np.sin(x + phase)
        population.append(alpha)

    # Random walk (20%)
    n_walk = int(pop_size * 0.2)
    for _ in range(n_walk):
        steps = np.random.normal(0, 1.5, n_stations)
        alpha = np.cumsum(steps) * 0.5
        alpha = gaussian_filter1d(alpha, sigma=3)
        alpha = np.clip(alpha, lower, upper)
        population.append(alpha)

    # Near centerline (10%)
    n_center = int(pop_size * 0.1)
    for _ in range(n_center):
        alpha = np.random.normal(0, 0.5, n_stations)
        alpha = gaussian_filter1d(alpha, sigma=2)
        population.append(alpha)

    # Fill remainder with pure random
    while len(population) < pop_size:
        alpha = np.random.uniform(lower, upper, n_stations)
        population.append(alpha)

    return population


def tournament_selection(population, fitness_values, tournament_size=5):
    selected = []
    pop_size = len(population)

    for _ in range(pop_size):
        indices = random.sample(range(pop_size), tournament_size)
        best_idx = max(indices, key=lambda i: fitness_values[i])
        selected.append(population[best_idx].copy())

    return selected


def blend_crossover(parent1, parent2):
    alpha = np.random.rand(len(parent1))
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2


def smooth_mutation(individual, mutation_rate, lower, upper, sigma):
    individual = np.array(individual, dtype=float).copy()
    n = len(individual)

    for i in range(n):
        if random.random() < mutation_rate:
            individual[i] += np.random.normal(0, sigma)

    individual = gaussian_filter1d(individual, sigma=2.0)
    return np.clip(individual, lower, upper)

