#Sydney Zheng
#Dr. Gabor
#Period 4
#11/16/15
#Sudoku Solver

import sys
from math import sqrt
import time

global digits
digits = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

global alls, size, mins
global digs, aGroups, peers, cellinGroup
global guess, known, deduce, guessed
guessed = []
global printlist
printlist = []

#assigns a value to a cell by removing that value from all of the cell's neighbors
def setFound(num, ind, assigned):
    assigned[ind] = num
    
    peergroup = peers[ind]
    for peer in peergroup:
        assigned[peer] = assigned[peer].replace(num, "")

    return assigned

#goes through the puzzle and eliminates possibilities in each cell
def prune(assigned):
    old = ""
    checked = set()
    while old != assigned:
        old = assigned[:]
        for ind in range(alls):
            if ind in checked:
                continue
            if len(assigned[ind]) == 1:
                num = assigned[ind]
                assigned = setFound(num, ind, assigned)
                checked.add(ind)
        
        if assigned == old:
            break
        
        for section in aGroups:
            for sym in digs:
                possible = [k for k in section if sym in assigned[k]]
                if len(possible) == 1:
                    assigned = setFound(sym, possible[0], assigned)
        
    return assigned

#attempts to assign a value to a cell by removing all the other possibilities from the cell
def edit(assigned, num, ind):
    othervals = assigned[ind].replace(num, "")#other values to remove
    if all(takeaway(assigned, n, ind) for n in othervals):
        return assigned
    return False

#attempts to remove a possibility from a cell and testing to see if it creates contradictions
def takeaway(assigned, num, ind):
    if not num in assigned[ind]: return assigned
    assigned[ind] = assigned[ind].replace(num, "")
    if len(assigned[ind]) == 0: return False
    
    if len(assigned[ind]) == 1:
        if not (all(takeaway(assigned, assigned[ind], i) for i in peers[ind])):
            return False#cannot be that num in ind because its peers need it as well
    #check for only in section
    for i in cellinGroup[ind]:
        sec = aGroups[i]
        #look for a place the eliminated num COULD be in each section
        couldbe = [j for j in sec if num in assigned[j]]
        if len(couldbe) == 0: return False#num has to be somewhere in the section
        if len(couldbe) == 1 and edit(assigned, num, couldbe[0]) == False:#edit should auto change assigned, if attempt to put the num in does not work, it fails
            return False
    return assigned

#guesses the value of a particular cell
def doGuess(assigned):
    global guess
    if assigned == False: return False
    if all(len(s) == 1 for s in assigned): return assigned
    ind = min((len(assigned[k]), k) for k in range(alls) if len(assigned[k]) > 1)[1]
    possible = list(assigned[ind])
    while len(possible) > 0:
        nassigned = doGuess(edit(assigned[:], possible.pop(), ind))
        if nassigned != False:#works
            guess += 1
            return nassigned
    return False

#creates the variable, assigned
def readfile(line):
    global alls, size, mins
    alls = len(line)
    size = int(sqrt(len(line)))
    mins = int(sqrt(size))
    
    global digs, aGroups, peers, cellinGroup
    digs = digits[0:size]
    aGroups = []
    peers = []
    cellinGroup = []
    
    global guess, known
    guess = 0
    known = 0
    
    assigned = []
    curr = ""
    
    for i in range(size*3):
        aGroups.append([])
        
    for ind in range(alls):
        rVal = int(ind/size)
        cVal = ind%size
        bVal = int(rVal / mins)*mins + int(cVal / mins)
        aGroups[bVal].append(ind)
        aGroups[size+rVal].append(ind)
        aGroups[2*size+cVal].append(ind)
        cellinGroup.append([bVal, size+rVal, 2*size+cVal])
            
        curr = line[ind: ind+1]
        if curr != ".":
            assigned.append(curr)
            known += 1
        else:
            assigned.append(digs)
    for ind in range(alls):
        rVal = int(ind/size)
        cVal = ind%size
        bVal = int(rVal / mins)*mins + int(cVal / mins)
        peers.append(list(set(aGroups[bVal]).union(set(aGroups[size+rVal]).union(set(aGroups[2*size+cVal])))))
        peers[ind].remove(ind)           
    return assigned

#prints Sudoku in 9x9 form
def printSudoku(assigned):
    for i in range(size):
        for j in range(size):
            if len(assigned[i*size+j]) == 1:
                sys.stdout.write(assigned[i*size+j] + " ")
            else:
                sys.stdout.write(". ")
            if (j+1)%mins == 0 and (j+1) != size and j != 0: #print col line
                sys.stdout.write("| ")
        print()
        if (i+1)%mins == 0 and (i+1) != size and i != 0: #print row line
            for i in range(size + mins-1):
                sys.stdout.write("--")
            print()
    print()

#determines if puzzle does not have repeat values in all sections
def validate(assigned):
    for grp in aGroups:
        listV = [assigned[ind] for ind in grp]
        if len(set(listV)) != len(listV): return False
    return True

#control method that solves a puzzle
def solve(line, puz):
    numstart = time.time()
    printlist.append("Puzzle " + str(puz) + "\n")
    printlist.append(line + "\n")
    
    assigned = readfile(line)
    assigned = prune(assigned)#do initial pruning of complete deduction
    assigned = doGuess(assigned)
         
    if assigned == False:
        printlist.append("Invalid Puzzle" + "\n")
    elif validate(assigned) == False:
        printlist.append("Invalid Solution" + "\n")
    else:
        guessed.append(guess)
        printlist.append("".join(assigned) + "\n")
    printlist.append(str(time.time()-numstart) + "\n")

#method that runs through each puzzle and prints appropriate figures
def main():
    sys.argv = [sys.argv[0], "puzzlesetsh.txt"]
    global printlist, guessed
    if len(sys.argv) < 2:
        print("Need filename")
    else:
        filename = sys.argv[1]
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
        
        start = time.time()
        if len(sys.argv) == 3:
            puz = int(sys.argv[2])
            line = lines[puz-1].strip()
            solve(line, puz)
            printlist.append(guess)
            
        elif (len(sys.argv) == 4) :
            num1 = int(sys.argv[2])
            num2 = int(sys.argv[3])
            if (num1 == num2):
                puz = num1
                line = lines[puz-1].strip()
                solve(line, puz)
                
            else:
                for puz in range(num1, num2+1):
                    line = lines[puz-1].strip()
                    solve(line, puz)
        else:#solve all puzzles
            for puz in range(len(lines)):
                line = lines[puz].strip()
                solve(line, puz+1)
                
        end = time.time()
        timelen = end - start
        print("".join(printlist))
        print("Guessed: " + str(sum(guessed)))
        print("Time: " + str(timelen))

main()
