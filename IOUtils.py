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


def dumpBoard(board):
	print( board[0] )
	print( board[1] )
	print( board[2] )
	print( board[3] )
	print( board[4] )
	print( board[5] )
	print( board[6] )
	print( board[7] )
	print( board[8] )
	print( board[9] )
	print()

def outputBoard(board):
	print( "    ABCDEFGHIJKLMNOPQRST")
	for i in range(mgame.DATA_GRID_H):
		print( "{0:2d}  ".format(i + 1), end = '')
		[ print(mgame.checkerToAscii(checker), end = '') for checker in board[i + 1] if checker != mgame.WALL]
		print()
	print()
	print()
