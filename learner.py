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


from sklearn.datasets.samples_generator import make_blobs
from numpy import loadtxt
from keras.optimizers import SGD 
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import main as mmain
import numpy

# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# Directions
none = 0
up = 1
down = 2
right = 3
left = 4

# Moves
Move_goStraight = 1
Move_turnRight = 2
Move_turnLeft = 3

#Checkers
EMPTY = 0
FRUIT = -1
POISON = -2
WALL = -3
SNAKE = 3
NECK = 2
HEAD = 1

#Checker sets
Good = (EMPTY, FRUIT)
Bad = (POISON, WALL, SNAKE, HEAD)


def mapDNNOutputVar(move):
    #print("[mapDNNOutputVar] Move = ", move)
    y = [ ]
    goStraight = False
    turnRight = False
    turnLeft = False
    if move == Move_goStraight:
        goStraight = True
    if move == Move_turnRight:
        turnRight = True
    if move == Move_turnLeft:
        turnLeft = True
    y = [ goStraight, turnRight, turnLeft ]
    #y = to_categorical(y)
    #print("[mapDNNOutputVar] y = ", y)
    return y

def mapDNNInputVar(game):
    fruitXRight = False
    fruitXLeft = False
    fruitXUp = False
    fruitXDown = False
    fruitQ1 = False
    fruitQ2 = False
    fruitQ3 = False
    fruitQ4 = False
    checkerUp = 0
    checkerDown = 0
    checkerRight = 0
    checkerLeft = 0

    if game.fruitY == game.snakeYHead:
        if game.fruitX > game.snakeXHead:
            fruitXRight = True
        else:
            fruitXLeft = True

    if game.fruitX == game.snakeXHead:
        if game.fruitY < game.snakeYHead:
            fruitXUp = True
        else:
            fruitXUp = False

    if game.fruitX > game.snakeXHead and game.fruitY < game.snakeYHead:
        fruitQ1 = True

    if game.fruitX < game.snakeXHead and game.fruitY < game.snakeYHead:
        fruitQ2 = True

    if game.fruitX < game.snakeXHead and game.fruitY > game.snakeYHead:
        fruitQ3 = True

    if game.fruitX > game.snakeXHead and game.fruitY > game.snakeYHead:
        fruitQ4 = True

    # --------------------------------------------
    checkerUpValue = game.currentBoard[game.snakeYHead - 1][game.snakeXHead]
    if checkerUpValue in Good:
        checkerUp = True
    else:
        checkerUp = False

    checkerDownValue = game.currentBoard[game.snakeYHead + 1][game.snakeXHead]
    if checkerDownValue in Good:
        checkerDown = True
    else:
        checkerDown = False

    checkerRightValue = game.currentBoard[game.snakeYHead][game.snakeXHead + 1]
    if checkerRightValue in Good:
        checkerRight = True
    else:
        checkerRight = False

    checkerLeftValue = game.currentBoard[game.snakeYHead][game.snakeXHead - 1]
    if checkerLeftValue in Good:
        checkerLeft = True
    else:
        checkerLeft = False

    # --------------------------------------------
    fruitD = game.distance( (game.snakeXHead, game.snakeYHead), (game.fruitX, game.fruitY) )
    
    if game.currentDirection == up:
        upD = game.distanceToFirstNonEmptyUp( (game.snakeXHead, game.snakeYHead) )
        downD = -1
        rightD = game.distanceToFirstNonEmptyRight( (game.snakeXHead, game.snakeYHead) )
        leftD = game.distanceToFirstNonEmptyLeft( (game.snakeXHead, game.snakeYHead) )

    if game.currentDirection == down:
        upD = -1
        downD = game.distanceToFirstNonEmptyDown( (game.snakeXHead, game.snakeYHead) )
        rightD = game.distanceToFirstNonEmptyRight( (game.snakeXHead, game.snakeYHead) )
        leftD = game.distanceToFirstNonEmptyLeft( (game.snakeXHead, game.snakeYHead) )

    if game.currentDirection == right:
        upD = game.distanceToFirstNonEmptyUp( (game.snakeXHead, game.snakeYHead) )
        downD = game.distanceToFirstNonEmptyDown( (game.snakeXHead, game.snakeYHead) )
        rightD = game.distanceToFirstNonEmptyRight( (game.snakeXHead, game.snakeYHead) )
        leftD = -1

    if game.currentDirection == left:
        upD = game.distanceToFirstNonEmptyUp( (game.snakeXHead, game.snakeYHead) )
        downD = game.distanceToFirstNonEmptyDown( (game.snakeXHead, game.snakeYHead) )
        rightD = -1
        leftD = game.distanceToFirstNonEmptyLeft( (game.snakeXHead, game.snakeYHead) )

    '''
    This returns all boolean values
    X = [ fruitXRight,
        fruitXLeft,
        fruitXUp,
        fruitXDown,
        fruitQ1,
        fruitQ2,
        fruitQ3,
        fruitQ4,
        checkerUp,
        checkerDown,
        checkerRight,
        checkerLeft ]
    '''
    # This returns 8 bool and 5 int
    X = [ fruitXRight,
        fruitXLeft,
        fruitXUp,
        fruitXDown,
        fruitQ1,
        fruitQ2,
        fruitQ3,
        fruitQ4,
        upD,
        downD,
        rightD,
        leftD,
        fruitD ]
    return X

def generateDNN(chromosome):
    print("[generateDNN] generating Neural Network...")
    X = chromosome[0]
    y = chromosome[1]
    learner = Learner()
    learner.Init()
    learner.train(X, y)
    return learner

class Learner:
    
    model = None

    def __init__(self):
        pass

    def HelloWorld(self):
        dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
        # split into input (X) and output (y) variables
        X = dataset[:,0:8]
        y = dataset[:,8]

        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(X, y, epochs=150, batch_size=10)

        W = model.get_weights()
        print("Model Weights = ", W)

    def Init(self):
        '''
        Model:

        Input: 13 vars as follow
        fruitYUp        bool, true if the fruit is in the same col as the snake's head, upper direction
        fruitYDown      bool, true if the fruit is in the same col as the snake's head, lower direction
        fruitXRight     bool, true if the fruit is in the same row as the snake's head, right direction
        fruitXLeft      bool, true if the fruit is in the same row as the snake's head, left direction
        fruitQ1         bool, true if the fruit is in quadrant 1 relative to the snake's head
        fruitQ2         bool, true if the fruit is in quadrant 2 relative to the snake's head
        fruitQ3         bool, true if the fruit is in quadrant 3 relative to the snake's head
        fruitQ4         bool, true if the fruit is in quadrant 4 relative to the snake's head
        upD		       	float holding the distance of a non empy checker from the snake's head in the up direction
        downD   		float holding the distance of a non empy checker from the snake's head in the down direction
        rightD    		float holding the distance of a non empy checker from the snake's head in the right direction
        leftD     		float holding the distance of a non empy checker from the snake's head in the left direction
		fruitD			float holding the distance of a the fruit from the snake's head
        
		Output: 3 vars holding the predicted direction (class) up, down, right, left
        goStraight:     bool, true is the prediction is to go straight
        turnRight:      bool, true is the prediction is to turn right
        turnLeft:       bool, true is the prediction is to turn left
        '''

        #opt = 'adam'
        #lss = 'binary_crossentropy'
        #actv = 'sigmoid'

        #opt = SGD(lr=0.01, momentum=0.9)
        #lss = 'categorical_crossentropy'
        #actv = 'softmax'
        
        # https://www.bmc.com/blogs/keras-neural-network-classification/
        opt = 'sgd'
        lss = 'binary_crossentropy'
        actv = 'sigmoid'
        
        self.model = Sequential()
        self.model.add(Dense(12, input_dim=13, activation='relu'))
        #self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(6, activation='relu'))
        self.model.add(Dense(3, activation=actv))
        self.model.compile(loss=lss, optimizer=opt, metrics=['accuracy'])


    def train(self, X, y):
        print("Size of X: ", len(X))
        #print("X = ", X)
        print("Size of y: ", len(y))
        #print("y = ", y)
        self.model.fit(numpy.array(X), numpy.array(y), epochs=len(X))

    def predictProbability(self, X):
        '''
        Not used.
        Returns an array containing the probabilities of the 4 classes
        '''
        sample = numpy.array([X])
        y = self.model.predict(sample)
        #print("Probabilities = ", y)
        return y

    def predictClass(self, X):
        #print("X = ", X)
        sample = numpy.array([X])
        y = self.model.predict_classes(sample)
        theClass = y[0]
        #print("class = ", theClass)
        return theClass

    def getWeights(self):
        W = self.model.get_weights()
        #print("Model Weights = ", W)
        #print("W len = ", len(W))
        #print("Model shape:")
        #[print(a.shape) for a in W]
        return W

    def setWeights(self, newWeights):
        self.model.set_weights(newWeights)
    
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    mmain.main()
