import Genetic
import random

geneSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

target = "x"

def generate_parent(length):

  genes = []

  while len(genes) < length: 

    sampleSize = min(length - len(genes), len(geneSet))

    genes.extend(random.sample(geneSet, sampleSize))

  return ''.join(genes)

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

def evolve():
  """ wrapper for genetic evolve """
  Genetic.evolve(get_fitness, len(target), len(target), geneSet, display)



def main(epochs):

  random.seed()

  sum = 0

  epoch = 0

  while epoch < epochs:

    bestGenome = generate_parent(len(target))

    fittest = get_fitness(bestGenome)

    display(bestGenome)

    generations = 0

    while True:

      generations += 1

      child = mutate(bestGenome)

      fitness = get_fitness(child)

      if fittest >= fitness:
        continue

      display(child)

      if fitness >= len(bestGenome):
        break

      fittest = fitness

      bestGenome = child

    sum += generations

    epoch += 1

  avg = sum / epochs

  return avg




