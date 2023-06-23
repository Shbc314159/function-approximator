import random
import neat_network
import random

class GeneticAlgorithm: 
    def __init__(self):
        self.population = []
        self.current_generation = 1
        self.num_moves = 20
        self.pop_size = 1000
        self.num_selected = 100
        self.mutation_probabilities = [0.5, 0.5, 0.5, 0.5, 0.5]
        self.best_individuals = []
            
    def create_population(self):
        for i in range(self.pop_size):
            nn = neat_network.neural_network(1, 1)
            nn.mutate(self.mutation_probabilities)
            self.population.append(nn) 
                
    def create_next_generation(self):
        self.best_individuals = self.selection()
        self.population = []

        total = 0
        for individual in self.best_individuals:
            total += individual.fitness
            individual.fitness = 0 
        print(total/self.num_selected) 
                
        for i in range(self.pop_size - len(self.best_individuals)):
            parent1 = random.choice(self.best_individuals)
            parent2 = random.choice(self.best_individuals)
            offspring_nn = parent1.crossover(parent2)
            offspring_nn.mutate(self.mutation_probabilities)
            self.population.append(offspring_nn)
            offspring_nn.fitness = 0
        
        for i in range(len(self.best_individuals)):
            parent = self.best_individuals[i]
            child_nn = parent.__copy__()
            self.population.append(child_nn)  
            child_nn.fitness = 0 
    
    def selection(self):
        selected_individuals = []
        sorted_individuals = sorted(self.population, key=lambda x: x.fitness)
        selected_individuals = sorted_individuals[:self.num_selected]
        
        return selected_individuals    