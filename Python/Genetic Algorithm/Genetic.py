def generate_parent(length, geneSet):

  genes = []

  while len(genes) < length: 

    sampleSize = min(length - len(genes), len(geneSet))

    genes.extend(random.sample(geneSet, sampleSize))

  return ''.join(genes)


def mutate(parent, geneSet):

  index = random.randrange(0, len(parent))

  child = list(parent)

  mutation, alternate = random.sample(geneSet, 2)

  child[index] = alternate \
      if mutation == child[index] \
      else mutation

  return ''.join(child)

def evolve(get_fitness, targetLength, goal, geneSet, display):

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