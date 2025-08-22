import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

# --- Hyperparameters for the Genetic Algorithm ---
POPULATION_SIZE = 5
GENERATIONS = 3
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

# --- MNIST Dataset ---
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)  # Add channel dimension
x_test = np.expand_dims(x_test, -1)


# --- Define CNN Architecture Genome ---
def create_individual():
    """Creates a random CNN architecture and hyperparameters."""
    individual = {
        'num_conv_layers': random.randint(1, 3),
        'filters': [random.choice([32, 64, 128]) for _ in range(3)],  # Max 3 layers
        'kernel_size': [random.choice([(3, 3), (5, 5)]) for _ in range(3)],  # Max 3 layers
        'pooling_size': [random.choice([(2, 2), (3, 3)]) for _ in range(3)],
        'dropout_rate': random.uniform(0.0, 0.5),
        'dense_units': random.choice([64, 128, 256]),
        'learning_rate': random.choice([0.001, 0.01, 0.1]),
        'batch_size': random.choice([32, 64, 128])
    }
    return individual


# --- Build and Train the CNN based on the Genome ---
def build_and_train_model(individual):
    """Builds, compiles, and trains a CNN based on the individual's genes."""
    model = Sequential()
    for i in range(individual['num_conv_layers']):
        model.add(Conv2D(individual['filters'][i], individual['kernel_size'][i], activation='relu', input_shape=(28, 28, 1)))  # Added input_shape
        model.add(MaxPooling2D(individual['pooling_size'][i]))
    model.add(Flatten())
    model.add(Dense(individual['dense_units'], activation='relu'))
    model.add(Dropout(individual['dropout_rate']))
    model.add(Dense(10, activation='softmax'))  # 10 classes for MNIST

    optimizer = Adam(learning_rate=individual['learning_rate'])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=1, batch_size=individual['batch_size'], verbose=0) # Reduced epochs for demonstration
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy

# --- Genetic Operators ---
def crossover(parent1, parent2):
    """Performs crossover between two parents to create two children."""
    if random.random() < CROSSOVER_RATE:
        child1 = {}
        child2 = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        return child1, child2
    else:
        return parent1, parent2


def mutate(individual):
    """Mutates an individual with a certain probability."""
    for key in individual.keys():
        if random.random() < MUTATION_RATE:
            if key == 'num_conv_layers':
                individual[key] = random.randint(1, 3)
            elif key == 'filters':
                individual[key] = [random.choice([32, 64, 128]) for _ in range(3)]
            elif key == 'kernel_size':
                individual[key] = [random.choice([(3, 3), (5, 5)]) for _ in range(3)]
            elif key == 'pooling_size':
                individual[key] = [random.choice([(2, 2), (3, 3)]) for _ in range(3)]
            elif key == 'dropout_rate':
                individual[key] = random.uniform(0.0, 0.5)
            elif key == 'dense_units':
                individual[key] = random.choice([64, 128, 256])
            elif key == 'learning_rate':
                individual[key] = random.choice([0.001, 0.01, 0.1])
            elif key == 'batch_size':
                individual[key] = random.choice([32, 64, 128])
    return individual


def selection(population, fitnesses, num_parents):
    """Selects the best individuals based on their fitness."""
    # Convert fitnesses to numpy array to use argsort
    fitnesses = np.array(fitnesses)
    indices = np.argsort(fitnesses)[-num_parents:]  # Indices of the best individuals
    parents = [population[i] for i in indices]
    return parents


# --- Main Genetic Algorithm ---
def genetic_algorithm():
    """Implements the genetic algorithm to evolve CNN architectures."""
    population = [create_individual() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        print(f"--- Generation {generation + 1} ---")

        # Evaluate Fitness
        fitnesses = [build_and_train_model(individual) for individual in population]
        for i in range(len(population)):
            print(f"Individual {i+1}: Accuracy = {fitnesses[i]:.4f}, Architecture = {population[i]}")

        # Selection
        parents = selection(population, fitnesses, POPULATION_SIZE // 2)

        # Crossover and Mutation to create new population
        new_population = parents.copy()
        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population  # Update the population

    # Find the best individual in the final population
    fitnesses = [build_and_train_model(individual) for individual in population]
    best_index = np.argmax(fitnesses)
    best_individual = population[best_index]
    best_accuracy = fitnesses[best_index]

    print("\n--- Final Result ---")
    print(f"Best Individual: Accuracy = {best_accuracy:.4f}, Architecture = {best_individual}")
    return best_individual, best_accuracy



if __name__ == "__main__":
    best_cnn, best_acc = genetic_algorithm()
    print("Genetic algorithm completed. Best CNN architecture found.")