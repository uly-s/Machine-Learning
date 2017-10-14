import Genetic
import random

geneSet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.!? 0123456789"

target = "Cedar"

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

def main():

  random.seed()

  bestGenome = generate_parent(len(target))

  fittest = get_fitness(bestGenome)

  display(bestGenome)

  while True:

    child = mutate(bestGenome)

    fitness = get_fitness(child)

    if fittest >= fitness:
      continue

    display(child)

    if fitness >= len(bestGenome):
      break

    fittest = fitness

    bestGenome = child



