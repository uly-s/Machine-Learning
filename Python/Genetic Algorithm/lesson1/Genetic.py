import random
import datetime

class Chromosome:

  def __init__(self, Genes, Fitness):

    self.Genes = Genes

    self.Fitness = Fitness
    

def generate_parent(length, geneSet):
  """ generates a length parent out of geneSet """

  genes = []

  while len(genes) < length: 

    sampleSize = min(length - len(genes), len(geneSet))

    genes.extend(random.sample(geneSet, sampleSize))

  return ''.join(genes)


def mutate(parent, geneSet):
  """Takes in a parent, makes a change in one gene, returns the child"""

  index = random.randrange(0, len(parent))

  child = list(parent)

  mutation, alternate = random.sample(geneSet, 2)

  child[index] = alternate \
      if mutation == child[index] \
      else mutation

  return ''.join(child)

def display(genes, target, startTime):
  """Generalized display function, pass a genome, the target, and a datetime object of start time"""

  timeDiff = datetime.datetime.now() - startTime

  fitness = get_fitness(genes, target)

  print("{}\t{}\t{}".format(genes, target, timeDiff))


def get_fitness(genes, target):
  """Returns number of of genes in genes that match the target"""
  return sum(1 for expected, actual in zip(target, guess) if expected == actual)


def evolve(get_fitness, targetLength, goal, geneSet, display):
  """ pass the fitness function, target length, desired fitness, the set of genes, and the display function """

  random.seed()

  bestGenome = generate_parent(targetLength, geneSet)

  fittest = get_fitness(bestGenome)

  display(bestGenome)

  while True:

    child = mutate(bestGenome, geneSet)

    fitness = get_fitness(child)

    if fittest >= fitness:
      continue

    display(child)

    if fitness >= len(bestGenome):
      break

    fittest = fitness

    bestGenome = child