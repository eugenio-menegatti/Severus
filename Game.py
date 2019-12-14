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


import copy
import main as mmain
from random import randrange
import IOUtils as mioutils
import learner as mlearner
import math

LOOP_WATCH_DOG = 10

STARTING_X = 5
STARTING_Y = 5

DATA_GRID_W = 20
DATA_GRID_H = 10

MoveType_random = 1
MoveType_predict = 2

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

moveMnemonics = { Move_goStraight: "goStraight", Move_turnRight: "turnRight", Move_turnLeft: "turnLeft" }

o = '.'		# empty
F = 'F' 	# Fruit
P = 'P' 	# Poison
W = 'W'		# Wall
S = 'O'     # Snake Body
N = 'O'     # Snake neck
H = '@'     # Sanke Head

#Checker
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

startingBoard = [
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W], 
		[W, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, W],
        [W, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, W],
        [W, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, W],
        [W, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, W],
        [W, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, W],
        [W, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, W],
        [W, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, W],
        [W, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, W],
        [W, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, W],
        [W, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, o, W],
        [W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W, W]
	]



def snakeCheckerGenerator():
    count = HEAD
    while True:
        yield count
        count += 1


def asciiToChecker(ascii):
    if ascii == o:
        return EMPTY
    if ascii == F:
        return FRUIT
    if ascii == P:
        return POISON
    if ascii == W:
        return WALL
    if ascii == S:
        return SNAKE
    if ascii == N:
        return SNAKE
    if ascii == H:
        return HEAD

def checkerToAscii(checker):
    if checker == EMPTY:
        return o
    if checker == FRUIT:
        return F
    if checker == POISON:
        return P
    if checker == WALL:
        return W
    if checker >= NECK:
        return S
    if checker == HEAD:
        return H


class Game:

    currentBoard = []
    snakeXHead = None
    snakeYHead = None
    snakeXTail = None
    snakeYTail = None
    fruitX = None
    fruitY = None
    score = None
    currentDirection = None
    moveCount = None
    looped = False

    def __init__(self):
        self.init()

    def init(self):
        self.currentBoard = copy.deepcopy(startingBoard)
        for y, row in enumerate(startingBoard):
            for x, ascii in enumerate(row):
                self.currentBoard[y][x] = asciiToChecker(ascii)

        self.snakeXHead = -1
        self.snakeYHead = -1
        self.snakeXTail = -1
        self.snakeYTail = -1
        self.fruitX = -1
        self.fruitY = -1
        self.currentDirection = up
        self.score = 0
        self.moveCount = 0
        self.looped = False

    def distance(self, p1, p2):
        x1 = p1[0]
        y1 = p1[1]
        x2 = p2[0]
        y2 = p2[1]
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        d = math.sqrt( dx * dx + dy * dy)
        return d
    
    def distanceToFirstNonEmptyRight(self, p):
        x = p[0]
        y = p[1]
        x0 = x
        d = 0
        while True:
            x += 1
            if self.currentBoard[y][x] != EMPTY:
                d = x - x0
                break
        return d

    def distanceToFirstNonEmptyLeft(self, p):
        x = p[0]
        y = p[1]
        x0 = x
        d = 0
        while True:
            x -= 1
            if self.currentBoard[y][x] != EMPTY:
                d = x0 - x
                break
        return d

    def distanceToFirstNonEmptyUp(self, p):
        x = p[0]
        y = p[1]
        y0 = y
        d = 0
        while True:
            y -= 1
            if self.currentBoard[y][x] != EMPTY:
                d = y0 - y
                break
        return d
    
    def distanceToFirstNonEmptyDown(self, p):
        x = p[0]
        y = p[1]
        y0 = y
        d = 0
        while True:
            y += 1
            if self.currentBoard[y][x] != EMPTY:
                d = y - y0
                break
        return d

    def respawnSnake(self):
        self.snakeXHead = STARTING_X
        self.snakeYHead = STARTING_Y
        self.snakeXTail = STARTING_X
        self.snakeYTail = STARTING_Y + 2
        x = self.snakeXHead
        y = self.snakeYHead
        
        snakeCheckerSequence = snakeCheckerGenerator()
        self.currentBoard[y + 0][x] = next(snakeCheckerSequence)
        self.currentBoard[y + 1][x] = next(snakeCheckerSequence)
        self.currentBoard[y + 2][x] = next(snakeCheckerSequence)

    def respawnFruit(self):
        respawned = False
        wd = 0
        while not respawned:
            x = randrange(DATA_GRID_W)
            y = randrange(DATA_GRID_H)
            if self.currentBoard[y][x] == EMPTY:
                self.fruitX = x
                self.fruitY = y
                self.currentBoard[y][x] = FRUIT
                respawned = True
            wd += 1
            if wd > 100:
                break
        return respawned

    def snakeDies(self):
        pass

    def snakeGoRight(self):
        canGo, what = self.canGoRight()
        if canGo:
            if what == FRUIT:
                self.score += 1
                self.growRight()
                self.respawnFruit()
            else:
                self.moveRight()
        else:
            self.snakeDies()
        return canGo, what

    def snakeGoLeft(self):
        canGo, what = self.canGoLeft()
        if canGo:
            if what == FRUIT:
                self.score += 1
                self.growLeft()
                self.respawnFruit()
            else:
                self.moveLeft()
        else:
            self.snakeDies()
        return canGo, what
    
    def snakeGoUp(self):
        canGo, what = self.canGoUp()
        if canGo:
            if what == FRUIT:
                self.score += 1
                self.growUp()
                self.respawnFruit()
            else:
                self.moveUp()
        else:
            self.snakeDies()
        return canGo, what

    def snakeGoDown(self):
        canGo, what = self.canGoDown()
        if canGo:
            if what == FRUIT:
                self.score += 1
                self.growDown()
                self.respawnFruit()
            else:
                self.moveDown()
        else:
            self.snakeDies()
        return canGo, what

    def canGoRight(self):
        x = self.snakeXHead
        y = self.snakeYHead
        if self.currentBoard[y][x + 1] in Good:
            return True, self.currentBoard[y][x + 1]
        else:
            return False, self.currentBoard[y][x + 1]

    def canGoLeft(self):
        x = self.snakeXHead
        y = self.snakeYHead
        if self.currentBoard[y][x - 1] in Good:
            return True, self.currentBoard[y][x - 1]
        else:
            return False, self.currentBoard[y][x - 1]

    def canGoUp(self):
        x = self.snakeXHead
        y = self.snakeYHead
        if self.currentBoard[y - 1][x] in Good:
            return True, self.currentBoard[y - 1][x]
        else:
            return False, self.currentBoard[y - 1][x]

    def canGoDown(self):
        x = self.snakeXHead
        y = self.snakeYHead
        if self.currentBoard[y + 1][x] in Good:
            return True, self.currentBoard[y + 1][x]
        else:
            return False, self.currentBoard[y + 1][x]

    def moveRight(self):
        x = self.snakeXHead
        y = self.snakeYHead 
        self.currentBoard[y][x + 1] = HEAD
        self.snakeXHead = x + 1
        self.moveSnake(x + 1, y, HEAD)
    
    def moveLeft(self):
        x = self.snakeXHead
        y = self.snakeYHead 
        self.currentBoard[y][x - 1] = HEAD
        self.snakeXHead = x - 1
        self.moveSnake(x - 1, y, HEAD)
    
    def moveUp(self):
        x = self.snakeXHead
        y = self.snakeYHead 
        self.currentBoard[y - 1][x] = HEAD
        self.snakeYHead = y - 1
        self.moveSnake(x, y - 1, HEAD)
    
    def moveDown(self):
        x = self.snakeXHead
        y = self.snakeYHead 
        self.currentBoard[y + 1][x] = HEAD
        self.snakeYHead = y + 1
        self.moveSnake(x, y + 1, HEAD)
    
    def moveSnake(self, x, y, snake):
        if self.currentBoard[y][x-1] == snake:
            if y == self.snakeYTail and x-1 == self.snakeXTail:
                self.currentBoard[y][x-1] = EMPTY
                self.snakeXTail = x
                return
            
            self.currentBoard[y][x-1] = snake + 1
            self.moveSnake(x-1, y, snake+1)
            return
        
        if self.currentBoard[y][x+1] == snake:
            if y == self.snakeYTail and x+1 == self.snakeXTail:
                self.currentBoard[y][x+1] = EMPTY
                self.snakeXTail = x
                return
            
            self.currentBoard[y][x+1] = snake + 1
            self.moveSnake(x+1, y, snake+1)
            return
        
        if self.currentBoard[y-1][x] == snake:
            if y-1 == self.snakeYTail and x == self.snakeXTail:
                self.currentBoard[y-1][x] = EMPTY
                self.snakeYTail = y
                return
            
            self.currentBoard[y-1][x] = snake + 1
            self.moveSnake(x, y-1, snake+1)
            return
        
        if self.currentBoard[y+1][x] == snake:
            if y+1 == self.snakeYTail and x == self.snakeXTail:
                self.currentBoard[y+1][x] = EMPTY
                self.snakeYTail = y
                return
            
            self.currentBoard[y+1][x] = snake + 1
            self.moveSnake(x, y+1, snake+1)
            return
        
    def growRight(self):
        x = self.snakeXHead
        y = self.snakeYHead 
        self.currentBoard[y][x + 1] = HEAD
        self.snakeXHead = x + 1
        self.growSnake(x + 1, y, HEAD)
    
    def growLeft(self):
        x = self.snakeXHead
        y = self.snakeYHead 
        self.currentBoard[y][x - 1] = HEAD
        self.snakeXHead = x - 1
        self.growSnake(x - 1, y, HEAD)
    
    def growUp(self):
        x = self.snakeXHead
        y = self.snakeYHead 
        self.currentBoard[y - 1][x] = HEAD
        self.snakeYHead = y - 1
        self.growSnake(x, y - 1, HEAD)
    
    def growDown(self):
        x = self.snakeXHead
        y = self.snakeYHead 
        self.currentBoard[y + 1][x] = HEAD
        self.snakeYHead = y + 1
        self.growSnake(x, y + 1, HEAD)

    def growSnake(self, x, y, snake):
        if self.currentBoard[y][x-1] == snake:
            if y == self.snakeYTail and x-1 == self.snakeXTail:
                self.currentBoard[y][x-1] = snake + 1
                return
            
            self.currentBoard[y][x-1] = snake + 1
            self.growSnake(x-1, y, snake+1)
            return
        
        if self.currentBoard[y][x+1] == snake :
            if y == self.snakeYTail and x+1 == self.snakeXTail:
                self.currentBoard[y][x+1] = snake + 1
                return
            
            self.currentBoard[y][x+1] = snake + 1
            self.growSnake(x+1, y, snake+1)
            return
        
        if self.currentBoard[y-1][x] == snake :
            if y-1 == self.snakeYTail and x == self.snakeXTail:
                self.currentBoard[y-1][x] = snake + 1
                return
            
            self.currentBoard[y-1][x] = snake + 1
            self.growSnake(x, y-1, snake+1)
            return
        
        if self.currentBoard[y+1][x] == snake :
            if y+1 == self.snakeYTail and x == self.snakeXTail:
                self.currentBoard[y+1][x] = snake + 1
                return
            
            self.currentBoard[y+1][x] = snake + 1
            self.growSnake(x, y+1, snake+1)
            return

    def isMoveValid(self, move):
        if move == none:
            return False
        return True

    def getPredictedMove(self, X, learner):
        move = none
        theClass = learner.predictClass(X)
        if theClass == 0: move = Move_goStraight
        if theClass == 1: move = Move_turnRight
        if theClass == 2: move = Move_turnLeft
        return move

    def getRandomMove(self):
        move = none
        rnd = randrange(3)
        if rnd == 0:
            move = Move_goStraight
        if rnd == 1:
            move = Move_turnRight
        if rnd == 2:
            move = Move_turnLeft
        return move

    def checkIfLooping(self, move):
        self.moveCount += 1
        if self.moveCount > DATA_GRID_W * DATA_GRID_H:
            return True
        else:
            return False

    def playOneMatch_Random(self, verbose):
        return self.playOneMatchImpl(MoveType_random, None, verbose)

    def playOneMatch_Predict(self, learner, verbose):
        #verbose = True
        return self.playOneMatchImpl(MoveType_predict, learner, verbose)

    def playOneMatchImpl(self, decisionType, learner, verbose):
        self.score = 0
        gameSituations = []
        gameDecisions = []
        self.respawnSnake()
        self.respawnFruit()
        gameOver = False
        while not gameOver:
            if verbose: mioutils.outputBoard(self.currentBoard)
            sample = mlearner.mapDNNInputVar(self)
            wd = 0
            moveIsValid = False
            while True:
                if decisionType == MoveType_random: move = self.getRandomMove()
                if decisionType == MoveType_predict:
                    move = self.getPredictedMove(sample, learner)
                    #print("Predicted move is: ", moveMnemonics[move])
                if self.isMoveValid(move):
                    moveIsValid = True
                    break
                if wd >= LOOP_WATCH_DOG:
                    break
                wd += 1
            if not moveIsValid:
                print("Could't evaluate a valid move - this Individual is quitting the match")
                gameOver = True
            else:
                looping = self.checkIfLooping(move)
                if looping:
                    #print("Player is looping - this Individual is forced to quit the match")
                    self.looped = True
                    gameOver = True
                else:
                    gameSituations.append(sample)
                    gameOver = self.snakeProceed(move)
                    decision = mlearner.mapDNNOutputVar(move)
                    #print("Move = ", move , "; Decision: ", decision)
                    gameDecisions.append(decision)
            if gameOver:
                if verbose: mioutils.outputBoard(self.currentBoard)
        return gameSituations, gameDecisions, self.score

    def mapMoveToDirection(self, move):
        direction = none
        if move == Move_goStraight:
            direction = self.currentDirection
        if move == Move_turnRight:
            if self.currentDirection == up:
                direction = right
            if self.currentDirection == down:
                direction = left
            if self.currentDirection == right:
                direction = down
            if self.currentDirection == left:
                direction = up
        if move == Move_turnLeft:
            if self.currentDirection == up:
                direction = left
            if self.currentDirection == down:
                direction = right
            if self.currentDirection == right:
                direction = up
            if self.currentDirection == left:
                direction = down
        return direction

    def snakeProceed(self, move):
        direction = self.mapMoveToDirection(move)
        self.currentDirection = direction

        if direction == right:
            canGo, what = self.snakeGoRight()
            if not canGo:
                if what in Bad or what > NECK:
                    return True
            return False
        if direction == left:
            canGo, what = self.snakeGoLeft()
            if not canGo:
                if what in Bad or what > NECK:
                    return True
            return False
        if direction == up:
            canGo, what = self.snakeGoUp()
            if not canGo:
                if what in Bad or what > NECK:
                    return True
            return False
        if direction == down:
            canGo, what = self.snakeGoDown()
            if not canGo:
                if what in Bad or what > NECK:
                    return True
            return False

def getSnakeDirection(self):
    canGo, what = self.canGoRight()
    if not canGo and what == NECK:
        return left
    
    canGo, what = self.canGoLeft()
    if not canGo and what == NECK:
        return right

    canGo, what = self.canGoUp()
    if not canGo and what == NECK:
        return down

    canGo, what = self.canGoDown()
    if not canGo and what == NECK:
        return up


#-------------------------------------------------------------------------------

if __name__ == "__main__":
    mmain.main()
