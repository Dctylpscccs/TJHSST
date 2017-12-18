import sys
import time
import random
global neighbors
global mapping
init = "...........................OX......XO..........................."
trans = ["I", "RL", "RR", "R2", "FX", "FY", "FD", "F0"]
#penalized for invalid moves

####SETUP####

def createNeighbors():
    global neighbors
    #create 2d array
    #make set from
    neighbors = {}
    n = set()
    l = r = t = b = -1
    for i in range(0, 8):
        if i == 0:t = -1
        else:t = i-1
        if i == 7:b = -1
        else:b = i+1
        for j in range(0, 8):
            if j == 0:l = -1
            else:l = j-1
            if j == 7:r = -1
            else:r = j+1
            n = set()
            if t != -1:n.add(t*8+j)
            if b != -1:n.add(b*8+j)
            if l != -1:
                n.add(i*8+l)
                if t != -1:n.add(t*8+l)
                if b != -1:n.add(b*8+l)
            if r != -1:
                n.add(i*8+r)
                if t != -1:n.add(t*8+r)
                if b != -1:n.add(b*8+r)
            neighbors[i*8+j] = n
    #done

####TRANSFORMATIONS####
def Map_I(i,j):return i, j
def Map_RL(i,j):return (8-1)-j, i
def Map_RR(i,j):return j, (8-1)-i
def Map_R2(i,j):return (8-1)-i, (8-1)-j
def Map_FX(i,j):return (8-1)-i, j
def Map_FY(i,j):return i, (8-1)-j
def Map_FD(i,j):return j, i
def Map_F0(i,j):return (8-1)-j, (8-1)-i
function_mappings = {
    "I": Map_I, "RL": Map_RL, "RR": Map_RR, "R2": Map_R2, "FX": Map_FX, "FY": Map_FY, "FD": Map_FD, "F0": Map_F0,
}

def mapA2B(key):
    IndMap = {}
    for i in range(0, 8):#row
        for j in range(0, 8):#col
            a, b = function_mappings[key](i, j)
            IndMap[a*8+b] = i*8+j
    return IndMap        
    
def createTrans():
    global mapping
    mapping = {}
    mapping["I"] = mapA2B("I")
    mapping["RL"] = mapA2B("RL")
    mapping["RR"] = mapA2B("RR")
    mapping["R2"] = mapA2B("R2")
    mapping["FX"] = mapA2B("FX")
    mapping["FY"] = mapA2B("FY")
    mapping["FD"] = mapA2B("FD")
    mapping["F0"] = mapA2B("F0")

####END SETUP####

def printBoard(string):
    for i in range(0, 9):
        if i == 8: #print out bottom row of column indices
            sys.stdout.write("  ")
            for j in range(0, 8):
                sys.stdout.write(str(j) + " ")
            continue
        sys.stdout.write(str(i) + " ")
        for j in range(0, 8):
            sys.stdout.write(string[i*8+j:i*8+j+1] + " ")
        sys.stdout.write(str(i) + "\n")
    print()
    #done

def recons(board, key):
    IndMap = mapping[key]
    newboard = ""
    for i in range(0, 64):
        newboard += board[IndMap[i]]
    return newboard

def checkPath(board, ind, done, dx, dy, turn, invt):
    y = int(ind/8)
    x = ind%8
    mid = False
    x = x+dx
    y = y+dy
    curr = y*8+x
    while (x >= 0 and x < 8) and (y >= 0 and y < 8) and board[curr:curr+1] == invt:
        mid = True
        x = x+dx
        y = y+dy
        curr = y*8+x
    if mid == True and (x >= 0 and x < 8) and (y >= 0 and y < 8) and board[curr:curr+1] == turn:
        return True
    return False
        
def findPossible(board, turn, invt):
    surround = set()
    done = set()
    for i in range(64):
        if board[i:i+1] == turn:
            done.add(i)
        if board[i:i+1] == invt:
            done.add(i)
            surround = surround.union(neighbors[i])
    surround = set([ind for ind in surround if ind not in done])#true surround
    #print(surround)
    itera = surround.copy()
    for ind in itera:
        good = False
        for dx in range(-1, 2):#-1,0,1
            for dy in range(-1, 2):#-1,0,1
                if dx == dy == 0:continue#at center
                via = checkPath(board, ind, done, dx, dy, turn, invt)
                if via == True:
                    good = True
                    break
        if good == False: surround.remove(ind)
    return surround, done
    #done

#MODERATOR FUNCTIONS
#for humans, run findPossible for them to see if they should pass
#print human and computer move then board with new string
#def validateMove():

#def executeMove(): #involves validating then flipping
def flipPath(board, ind, done, dx, dy, turn, invt):
    if checkPath(board, ind, done, dx, dy, turn, invt) == False:
        return board#don't do anything
    y = int(ind/8)
    x = ind%8
    x = x+dx
    y = y+dy
    curr = y*8+x
    while (x >= 0 and x < 8) and (y >= 0 and y < 8) and board[curr:curr+1] == invt:
        board = board[0:curr] + turn + board[curr+1:]
        x = x+dx
        y = y+dy
        curr = y*8+x
    return board

def flip(board, move, done, turn, invt):
    #need to flip over wherever possible
    for dx in range(-1, 2):#-1,0,1
        for dy in range(-1, 2):#-1,0,1
            if dx == dy == 0:continue#at center
            board = flipPath(board, move, done, dx, dy, turn, invt)
    return board

def HumTurn(board, turn, invt):
    surround, done = findPossible(board, turn, invt)
    if len(surround) == 0:
        print("Player " + turn + " Must Pass")
        return board, True
    print("Available: " + str(surround))
    print("Player " + turn + " Print Move: ")
    
    move = -1
    parts = []
    while move not in surround:
        given = input()
        parts = given.split(" ")#split by a space
        if parts[0].upper() in trans:
            newboard = recons(board[0:], parts[0].upper())
            printBoard(newboard)
            continue#find move in while loop
        if len(parts) < 2:
            move = int(parts[0])
        else:
            move = int(parts[0])*8 + int(parts[1])
        if move not in surround:
            print("Invalid Move")
    board = board[0:move] + turn + board[move+1:]#switch turn
    board = flip(board, move, done, turn, invt)
    return board, False
        
    
def CompTurn(board, turn, invt):
    surround, done = findPossible(board, turn, invt)
    if len(surround) == 0:
        print("Computer Passes")
        return board, True
    move = random.sample(surround, 1).pop()#pick a random move for now
    #TO EXPAND
    board = board[0:move] + turn + board[move+1:]#switch turn
    board = flip(board, move, done, turn, invt)
    return board, False

def score(board):
    X = 0
    O = 0
    for i in range(64):
        if board[i:i+1] == "X": X+=1
        if board[i:i+1] == "O": O+=1
        #if "." then do nothing
    print(str(X) + "-" + str(O))
    if X > O:
        print("X Wins!")
    elif O > X:
        print("O Wins!")
    else: #tie
        print("Tie")
    
def moderator(P1, P2):#first is true if human is first
    print("OTHELLO\n")
    board = init
    printBoard(board)
    end = False
    turn = True#first
    p1 = False
    p2 = False
    while end == False:
        if turn == True:
            if P1 == True:
                board, p1 = HumTurn(board, "X", "O")#p1 is pass or not
            else:
                board, p1 = CompTurn(board, "X", "O")
        else:
            if P2 == True:
                board, p2 = HumTurn(board, "O", "X")#p2 is pass or not
            else:
                board, p2 = CompTurn(board, "O", "X")
        printBoard(board)#2d version
        turn = not turn#switch player
        end = (p1 == True and p2 == True)
    "Game Over"
    score(board)

#for the purpose of creating a game
def central():
    createNeighbors()
    createTrans()
    sys.argv.append("2")
    if len(sys.argv) < 2:
        print("Need Input")
        quit()
    inp = sys.argv[1]
    P1 = False
    P2 = False
    if inp == "2": #hum v. hum
        P1 = P2 = True
    elif inp == "1C":
        P1 = True
        P2 = False
    elif inp == "C1":
        P1 = False
        P2 = True
    elif inp == "CC":
        P1 = P2 = False
    else:
        print("Unknown Input")
        quit()
    moderator(P1, P2)
    quit()
        
#for the purpose of the competition
def main():
    start = time.time()
    createNeighbors()
    #createTrans()
    sys.argv.append("...........................XO......OX...........................")
    sys.argv.append("X")
    inp1 = False
    inp2 = False
    if len(sys.argv) > 1:
        inp1 = sys.argv[1]
    if len(sys.argv) > 2:
        inp2 = sys.argv[2]
    if inp1 != False and len(inp1) == 64 and inp2 == False: #print board
        printBoard(inp1)
    if inp1 != False and len(inp1) == 64 and inp2 != False:
        print(neighbors)
        printBoard(inp1)
        if inp2 == "O": invt = "X"
        if inp2 == "X": invt = "O"
        surround, done = findPossible(inp1, inp2, invt)
        print()
        print(surround)
        print(time.time()-start)        

#TO RUN
central()
#main()