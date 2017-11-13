import Genetic
import random

geneSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .!?,"

target = "Hold your head up high"

def get_fitness(guess):

   return Genetic._get_fitness(guess, target)

def display(guess):

  fitness = get_fitness(guess)

  print('{} {}'.format(guess, fitness))

def evolve():
  """ wrapper for genetic evolve """
  return Genetic.evolve(get_fitness, len(target), len(target), geneSet, display)



def main():

  best = evolve()

  print(best.Genes)
  





