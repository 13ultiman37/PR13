import pandas as pd
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None,
                        halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


POPULATION_SIZE = 50
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.3  # probability for mutating an individual
MAX_GENERATIONS = 80
HALL_OF_FAME_SIZE = 5
FEATURE_PENALTY_FACTOR = 0.001

random.seed(42)


class Breasts():
    NUM_FOLDS = 5

    def __init__(self):
        self.data = pd.read_csv('breast-cancer-wisconsin1.data')

        self.X = self.data.iloc[:, 1:10]
        self.y = self.data.iloc[:, 10]

        self.kfold = model_selection.KFold(
            n_splits=self.NUM_FOLDS, random_state=443, shuffle=True)
        self.classifier = DecisionTreeClassifier(random_state=30)

    def __len__(self):
        return self.X.shape[1]

    def get_mean_accuracy(self, zeroOneList):
        zeroIndicies = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX = self.X.drop(self.X.columns[zeroIndicies], axis=1)

        cv_results = model_selection.cross_val_score(self.classifier,
                                                     currentX,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy')
        return cv_results.mean()


breasts = Breasts()

toolbox = base.Toolbox()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("zeroOrOne", random.randint, 0, 1)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(breasts))
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def breast_cancer_classification_accuracy(individual):
    numFeaturesUsed = sum(individual)
    if numFeaturesUsed == 0:
        return 0.0,
    else:
        accuracy = breasts.get_mean_accuracy(individual)
        return accuracy - FEATURE_PENALTY_FACTOR * numFeaturesUsed,


toolbox.register('evaluate', breast_cancer_classification_accuracy)
toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=1.0 / len(breasts))

population = toolbox.populationCreator(n=POPULATION_SIZE)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register('max', numpy.max)
stats.register('avg', numpy.mean)

hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

population, logbook = eaSimpleWithElitism(population=population,
                                          toolbox=toolbox,
                                          cxpb=P_CROSSOVER,
                                          mutpb=P_MUTATION,
                                          ngen=MAX_GENERATIONS,
                                          stats=stats,
                                          halloffame=hof,
                                          verbose=True)

print('\n\nЛучшие решения:')
for i in range(HALL_OF_FAME_SIZE):
    print(i, ': ', hof.items[i], ', приспособленность = ', hof.items[i].fitness.values[0],
          ', верность = ', breasts.get_mean_accuracy(hof.items[i]),
          ', признаков = ', sum(hof.items[i]))

maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

sns.set_style('whitegrid')
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Average Fitness')
plt.title('Max and Average fitness over Generations')
plt.show()

allOnes = [1] * len(breasts)
print('\n\nВыделены все признаки: ', allOnes, ', accuracy = ', breasts.get_mean_accuracy(allOnes))

diff = breasts.get_mean_accuracy(hof.items[0]) - breasts.get_mean_accuracy(allOnes)
print('\n\nВерность повысилась на {:.5f}'.format(diff))
