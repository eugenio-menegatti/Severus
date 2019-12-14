'''
SEVERUS - A neural network genetic training program
Copyright 2019 Eugenio Menegatti
myindievg@gmail.com

	 This file is part of SEVERUS.
	 The file COPYING describes the terms under which SEVERUS is distributed.

   SEVERUS is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   SEVERUS is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with SEVERUS.  If not, see <http://www.gnu.org/licenses/>.
 '''


import main as mmain
import learner as mlearner
import Game as mgame 
import numpy
import copy

MaxPopulation = 1000 # Number of individuals in each generation
MaxGeneration = 30 # Number of generations to breed, excluding the random starting generation

Population = []

def randomizePopulation():
    population = []
    for i in range(MaxPopulation):
        print("[ranzomizePopulation] Generating chromosome: ", i)
        game = mgame.Game()
        X, y, score = game.playOneMatch_Random(False)
        #print("X = ", X)
        #print("y = ", y)
        population.append([X, y, score])
    return population

def fitnessFunction(Population, i):
    chromosome = Population[i]
    score = chromosome[2]
    return score

def maximizeFitnessFunction(population, howMany):
    max = []
    for _ in range(howMany):
        maxI = -1
        maxScore = -999
        for i, _ in enumerate(population):
            score = fitnessFunction(population, i)
            #print(i, score)
            if score > maxScore and i not in max:
                maxScore = score
                maxI = i
        max.append(maxI)
    #print("max = ", max)
    return max

def sexual(population):
    #print("Population size: ", len(population))
    max = maximizeFitnessFunction(population, 2)
    motherI = max[0]
    fatherI = max[1]
    #print("Mother: ", motherI)
    #print("Father: ", fatherI)
    mother = population[motherI]
    father = population[fatherI]
    #print("Mother: ", mother)
    #print("Father: ", father)
    return mother, father

def clonation(population):
    max = maximizeFitnessFunction(population, 1)
    motherI = max[0]
    mother = population[motherI]
    return mother

def selection(population):
    print("[selection] Selecting fittest(s)")
    #return sexual(population)
    return clonation(population), None

def mutation():
    pass

def combineWeights(motherWeights, fatherWeights):
    # check if mother and father are compatible
    for i, a in enumerate(motherWeights):
        if a.shape != fatherWeights[i].shape:
            return None
    
    # entirely copy the mother onto the child so that it also gets the right shape
    childWeights = copy.deepcopy(motherWeights) #[numpy.copy(a) for a in motherWeights]
    
    #print("before")
    #[print(a.shape) for a in childWeights]

    # copy half of the father into the lower part of the child
    for i, a in enumerate(childWeights):
        if a.ndim == 2:
            #print("a is a matrix")
            h, w = a.shape
            h2 = int(h / 2)
            a[h2 : h, 0 : w] = fatherWeights[i][h2 : h, 0 : w]
        if a.ndim == 1:
            #print("a is a vector")
            l = a.size
            l2 = int(l / 2)
            a[l2 : l] = fatherWeights[i][l2 : l]
    
    #print("after")
    #[print(a.shape) for a in childWeights]
    return childWeights

def crossover(motherDNN, fatherDNN):
    print("[crossover] Generating child")
    motherWeights = motherDNN.getWeights()
    fatherWeights = fatherDNN.getWeights()
    childWeights = combineWeights(motherWeights, fatherWeights)
    child = mlearner.Learner()
    child.Init()
    child.setWeights(childWeights)
    return child

def breed(individual, generation):
    print("[breed] Creating population ", generation)
    population = []
    loopers = 0
    for i in range(MaxPopulation):
        if i % 100 == 0:
            print("[breed] Generating chromosome: ", i, " of generation ", generation)
        game = mgame.Game()
        X, y, score = game.playOneMatch_Predict(individual, False)
        if game.looped:
            loopers += 1
        population.append([X, y, score])
    if loopers != 0:
            print("Looping individuals in generation ", generation, ": ", loopers)
    return population

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    mmain.main()
