import Genetic
import random

geneSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.!? 0123456789"

target = "CedarLikeTheTree"

def get_fitness(guess):

  return sum(1 for expected, actual in zip(target, guess) if expected == actual)

def mutate(parent):

  index = random.randrange(0, len(parent))

  child = list(parent)

  mutation, alternate = random.sample(geneSet, 2)

  child[index] = alternate \
      if mutation == child[index] \
      else mutation

  return ''.join(child)

def display(guess):

  fitness = get_fitness(guess)

  print('{} {}'.format(guess, fitness))

def main():

  length = len(target)

  Genetic.evolve(get_fitness, length, length, geneSet, display)



