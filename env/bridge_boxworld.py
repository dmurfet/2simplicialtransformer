# Version for publication (v4)

import random

GOAL_TILE = 0

def displaceCell(cell, displacement, grid):
    return [min(max(cell[0]+displacement[0],0),len(grid)-1), min(max(cell[1]+displacement[1],0),len(grid[0])-1)]

def getCellType(cell, grid):
    return grid[min(max(cell[0],0),len(grid)-1)][min(max(cell[1],0),len(grid[0])-1)]

def clearCell(cell, grid):
    grid[min(max(cell[0],0),len(grid)-1)][min(max(cell[1],0),len(grid[0])-1)] = []

def tryOpenMultiBox(pInventory, keyType, keyOtherType, lockType, lockOtherType):
    #check if the player has all of the appropriate keys
    for lock in lockType:
        if lock not in pInventory:
            return False
            
    for lock in lockOtherType:
        if lock not in pInventory:
            return False
        
    #if they do, then remove all those keys from the inventory
    for lock in lockType:
        pInventory.remove(lock)
        
    for lock in lockOtherType:
        pInventory.remove(lock)
        
    #and add the keys inside the box to the inventory
    pInventory += keyType
    pInventory += keyOtherType

    return True
    
def tryOpenBox(pInventory, keyType, lockType):
    #check if the player has all of the appropriate keys
    for lock in lockType:
        if lock not in pInventory:
            return False
        
    #if they do, then remove all those keys from the inventory
    for lock in lockType:
        pInventory.remove(lock)
        
    #and add the keys inside the box to the inventory
    pInventory += keyType

    return True

def createColoursArray():
    colours = []
    for i in range(90, 210, 40):
        for j in range(90, 210, 40):
            for k in range(90, 210, 40):
                if( (i != 130 or j != 130 or k != 130)
                and (i != 210 or j != 210 or k != 210)
                and (i != j or i != k or j != k) ):
                    colours += [(i,j,k)]

    # randomise the order of the colours
    random.shuffle(colours)

    # set the goal tile colour as red
    colours[GOAL_TILE] = (255, 0, 0)

    return colours
    

def checkGridValid(grid):
    # check that the dimensions of the grid are valid
    #  (length > 0, all rows same length)
    numCols = len(grid)
    if (numCols <= 0):
        print("grid error: no columns specified")
        return False
    numRows = len(grid[0])
    if (numRows <= 0):
        print("grid error: no rows specified")
        return False
    for col in grid:
        if len(col) != numRows:
            print("grid error: columns had different lengths")
            return False

    # check that boxes do not occur on
    # (1) rows with an even index
    # (2) columns with an index divisible by three
    # this ensures that there is enough space to move the player around
    #  and also that we don't have three non-blank tiles adjacent to one another
    for i in range(0, numCols, 3):
        for j in range(0, numRows, 2):
            if grid[i][j] != []:
                print("grid error: box on invalid tile: ", i, j)
                return False

    return True
                
def generatePuzzle(numCols, numRows, solutionLength, numDecoyPaths, maxDecoyLength, multiLock, hasBridge):
    #generates a simple puzzle with no branching
    #TODO: improve this so it has branching
    grid = [[[]] * numRows for i in range(numCols)]
    decoys = []
    
    #determine which locations on the grid are valid box locations
    boxes = []
    for i in range(1, numCols-1, 3):
        for j in range(1, numRows, 2):
            boxes += [(i,j)]
    
    #shuffle the box locations
    random.shuffle(boxes)
    nextBox = 0
    
    if( multiLock ):
        while( (boxes[0])[0] > numCols - 3
                or (boxes[0])[1] > numRows - 3 ):
            random.shuffle(boxes)
            
        goal_box = boxes[0]
        box_below_goal = (goal_box[0],goal_box[1]+2)
        
        boxes.remove(box_below_goal)

    #populate the grid with boxes
    while nextBox < solutionLength:
        box = boxes[nextBox]
        grid[box[0]][box[1]] = [nextBox]
        grid[box[0]+1][box[1]] = [nextBox+1]
        nextBox += 1

    #place a key onto the grid which is not locked inside a box
    startingKey = boxes[nextBox]
    grid[startingKey[0]][startingKey[1]] = [nextBox]
    nextBox += 1
    nextColour = nextBox
    
    if( multiLock ):
        # adorn the original goal 
        goal_box = boxes[0]
        grid[goal_box[0]][goal_box[1]+1] = [0]
        grid[goal_box[0]+1][goal_box[1]+1] = [nextColour]
        
        # nextBox = solutionLength+1, ...., 2*solutionLength-1
        while nextBox < 2*solutionLength:
            box = boxes[nextBox]
            grid[box[0]][box[1]] = [nextColour]
            grid[box[0]+1][box[1]] = [nextColour+1]
            nextBox += 1
            nextColour += 1
        
        #place a key onto the grid which is not locked inside a box
        startingKey = boxes[nextBox]
        grid[startingKey[0]][startingKey[1]] = [nextColour]
        nextBox += 1
        nextColour += 1
        
    # Allocate bridges. There is a 50% chance an episode contains a bridge
    allocateBridge = random.randint(0,1)
    
    if( multiLock and hasBridge and allocateBridge == 0):
        bridgeSource = random.randint(1, solutionLength)
        bridgeTarget = random.randint(solutionLength+1, 2*solutionLength)
        
        bridgeBox = boxes[nextBox]
        grid[bridgeBox[0]][bridgeBox[1]] = [bridgeTarget]
        grid[bridgeBox[0]+1][bridgeBox[1]] = [bridgeSource]
        nextBox += 1
        
        decoys.append([bridgeBox[0]+1,bridgeBox[1]])
    
    #add some decoy paths
    # these are simply linear branches off the main solution path
    # ie. branches which do not themselves contain further branching
    for i in range(0, numDecoyPaths):
        #generate a random decoy path length
        pathLength = random.randint(1, maxDecoyLength)
        #generate the branching point
        
        if( not multiLock ):
            decoyLockType = random.randint(1, solutionLength)
        else:
            decoyLockType = random.randint(1, 2 * solutionLength)
        
        #generate the branch
        for j in range(0, pathLength):
            #get box location
            box = boxes[nextBox]
            #create a box whose lock is of the previously seen type
            
            if( j == 0 ):
                grid[box[0]+1][box[1]] = [decoyLockType]
            else:
                grid[box[0]+1][box[1]] = [nextColour-1]
                
            decoys.append([box[0]+1,box[1]])
            #generate the key inside the box
            
            grid[box[0]][box[1]] = [nextColour]
            nextBox += 1
            nextColour += 1
    
    return grid, decoys



def updateState(grid, pPos, pInventory, pInput, multiLock):
    # this function updates the game state based on the current state and player input
    # 
    # grid should be a list of lists of integers, containing the tile data
    #  grid[i][j] should be the tile in x-position i and y-position j
    #  where the origin is in the _upper_ left corner
    #  if grid[i][j] == 0, then the tile at (i,j) is blank
    #
    # pPos should be a two-entry list (x and y pos of player)
    #  note that the player position is not stored in the grid array.
    #
    # pInventory should be a list containing non-zero integers (keys currently in inventory)
    #
    # pInput should be a two-entry list, telling us which way the player wishes to move on this frame

    #get our target cell based on the current player position and the player input
    pTargetCell = displaceCell(pPos, pInput, grid)

    # check whether the target cell is empty; if so, move there
    if getCellType(pTargetCell, grid) == []:
        pPos = pTargetCell[:]
    else:
        # check the cell to the left and right to see if there's a lock
        cellL = displaceCell(pTargetCell, [-1,0], grid)
        cellR = displaceCell(pTargetCell, [1,0], grid)
        cellLType = getCellType(cellL, grid)
        cellMType = getCellType(pTargetCell, grid)
        cellRType = getCellType(cellR, grid)
        
        if( multiLock and cellLType == [0] ):
            pTargetCellUp = displaceCell(pTargetCell,[0,-1],grid)
            pTargetCellDown = displaceCell(pTargetCell,[0,1],grid)
            cellLUp = displaceCell(cellL, [0,-1], grid)
            cellLDown = displaceCell(cellL,[0,1],grid)
            
            if( getCellType(pTargetCellUp,grid) == [] ):
                pTargetCellOther = pTargetCellDown
            else:
                pTargetCellOther = pTargetCellUp
                
            if( getCellType(cellLUp,grid) == []):
                keyCellOther = cellLDown
            else:
                keyCellOther = cellLUp
                
            keyCellOtherType = getCellType(keyCellOther,grid)
            pTargetCellOtherType = getCellType(pTargetCellOther,grid)
            
            if tryOpenMultiBox(pInventory, cellLType, keyCellOtherType, cellMType, pTargetCellOtherType):
                clearCell(cellL,grid)
                clearCell(keyCellOther,grid)
                clearCell(pTargetCell,grid)
                clearCell(pTargetCellOther,grid)
                pPos = pTargetCell[:]
        else:
            if cellLType != []:
                #then the current cell is a lock, and the cell on its left is a key
                if tryOpenBox(pInventory, cellLType, cellMType):
                    clearCell(cellL, grid)
                    clearCell(pTargetCell, grid)
                    pPos = pTargetCell[:]

            elif cellRType == []:
                #then we have a key which is not inside a box; add it to inventory
                pInventory += cellMType
                clearCell(pTargetCell, grid)
                pPos = pTargetCell[:]
        

    return [grid, pPos, pInventory]


