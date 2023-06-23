from Genetic_Algorithm import GeneticAlgorithm
import random 

import gc
import cProfile

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
line_actual, = ax.plot([], [], color='blue', label='y = x^2')
line_network, = ax.plot([], [], color='red', label='Neural Network')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Function Approximation')
ax.legend()

def main(): 
    inputs = np.linspace(-10, 10, 100)
    outputs = []

    for element in inputs:
        outputs.append(element**2)
        
    genetic_algorithm = GeneticAlgorithm()    
    
    while genetic_algorithm.current_generation < 1000:
        print("Generation:", genetic_algorithm.current_generation)
        
        if genetic_algorithm.current_generation == 1:
            genetic_algorithm.create_population()
        else:
            genetic_algorithm.create_next_generation()  
        
        for individual in genetic_algorithm.population:
            
            for i in range(genetic_algorithm.num_moves):
                  
                random_index = random.randint(0, 99)    
                result = individual.run([inputs[random_index]])
                individual.fitness += abs(outputs[random_index]- result)
            
            individual.set_fitness()
        
        genetic_algorithm.current_generation += 1
        approximate_function(genetic_algorithm, inputs, outputs)

def approximate_function(ga, inputs, outputs):
    if len(ga.best_individuals) > 0:
        y_network = run_best(ga, inputs)

        line_actual.set_data(inputs, outputs)
        line_network.set_data(inputs, y_network)

        ax.relim()
        ax.autoscale_view(True, True, True)
        plt.draw()
        plt.pause(0.01)

def run_best(ga, x):
    individual = ga.best_individuals[0]
    array = []   
    for element in x:
        array.append(individual.run([element]))   
    return array
    
if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats('profile_results.prof')
   
        