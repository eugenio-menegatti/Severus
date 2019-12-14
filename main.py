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


 import Game as mgame
import IOUtils as mioutils 
import genetics as mgenetics
import learner as mlearner
import numpy

def main():
    print("Severus start")
    print("Randomizing population")
    population = mgenetics.randomizePopulation()
    print("Starting evolution")
    for g in range(mgenetics.MaxGeneration):
        generation = g + 1
        print("")
        print("Generation: ", generation, " - population size: ", len(population))
        motherChromosome, fatherChromosome = mgenetics.selection(population)
        print("Training mother")
        motherDNN = mlearner.generateDNN(motherChromosome)
        if fatherChromosome is not None:
            print("Training father")
            fatherDNN = mlearner.generateDNN(fatherChromosome)
            child = mgenetics.crossover(motherDNN, fatherDNN)
        else:
            child = motherDNN
        if generation < mgenetics.MaxGeneration-1: 
            population = mgenetics.breed(child, generation)
    
    finalIndividual = child
    print()
    input("Press ENTER to start Final-Individual's game...")
    game = mgame.Game()
    result = game.playOneMatch_Predict(finalIndividual, True)
    print("Score = ", result[2])
    print()
    print("Severus end")

def test():
    game = mgame.Game()
    mioutils.outputBoard(game.currentBoard)
    
    game.respawnSnake()
    mioutils.outputBoard(game.currentBoard)

    game.snakeGoRight()
    mioutils.outputBoard(game.currentBoard)

    game.growUp()
    mioutils.outputBoard(game.currentBoard)

if __name__ == "__main__":
    main()
    