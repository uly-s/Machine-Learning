import random
import datetime

class Chromosome:

  def __init__(self, Genes, Fitness):

    self.Genes = Genes

    self.Fitness = Fitness
    

def _generate_parent(length, geneSet, get_fitness):
  """ generates a length parent out of geneSet """

  genes = []

  while len(genes) < length: 

    sampleSize = min(length - len(genes), len(geneSet))

    genes.extend(random.sample(geneSet, sampleSize))

  fitness = get_fitness(genes)

  return Chromosome(genes, fitness)


def _mutate(parent, geneSet, get_fitness):
  """Takes in a parent, makes a change in one gene, returns the child"""

  index = random.randrange(0, len(parent.Genes))

  child = list(parent.Genes)

  mutation, alternate = random.sample(geneSet, 2)

  child[index] = alternate \
      if mutation == child[index] \
      else mutation

  fitness = get_fitness(child)

  return Chromosome(child, fitness)

def _display(genes, target, startTime, get_fitness):
  """Generalized display function, pass a genome, the target, and a datetime object of start time"""

  timeDiff = datetime.datetime.now() - startTime

  fitness = get_fitness(genes, target)

  print("{}\t{}\t{}".format(genes, target, timeDiff))


def _get_fitness(genes, target):
  """Returns number of of genes in genes that match the target"""
  return sum(1 for expected, actual in zip(target, genes) if expected == actual)


def evolve(get_fitness, targetLength, goal, geneSet, display):
  """ pass the fitness function, target length, desired fitness, the set of genes, and the display function """

  random.seed()

  bestGenome = _generate_parent(targetLength, geneSet, get_fitness)

  fittest = get_fitness(bestGenome.Genes)

  display(bestGenome.Genes)

  while True:

    child = _mutate(bestGenome, geneSet, get_fitness)

    if bestGenome.Fitness >= child.Fitness:
      continue

    display(child.Genes)

    if child.Fitness >= goal:
      return child

    bestGenome = child