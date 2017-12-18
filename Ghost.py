import sys
import time

#GETCH
class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()

class Node:
    def __init__(self, value, listW):
        self.value = value
        self.children = listW
    def size(self):
        return len(self.children)
    def value(self):
        return self.value
    def allChildren(self):
        return self.children
    
        
def createList():
    srcfile = open("ghost.txt", "r")
    wordList = []
    for line in srcfile:
        wordList.append(line[:len(line)-1])#remove newline char
    return wordList
    
def build(wordList):
    length = len(wordList) 
    if length == 1: #root node
        a = set()
        return a.add(Node(wordList[0], set()))
    ch1 = set()#at most 26 members for each letter of alphabet
    send = []
    i = 0
    j = 0
    while i != length:
        startLet = wordList[i][0:1]#first char of word
        while j < length and wordList[j][0:1] == startLet:
            send.append(wordList[j][1:])#everything but first letter
            j += 1
        ch2 = build(send)
        child = Node(startLet, ch2)
        ch1.add(child)
        send = []#reset
        i = j
    return ch1

#used when trying to determine possible set of chars to use or if it is a word or not
#if a particular node is needed to determine if it holds a word or a set of words, or if it doesn't exist
def traverse(head, givenstring):#givenstring is the current string, and helps find the last node
    children = head.allChildren()
    if len(givenstring) == 0:
        return children
    if len(children) == 1:
        c = children.pop()
        children = children.add(c)
        v = c.value()
        if v == givenstring:#a complete word
            return True
        else:
            return False
    for child in children:
        if child.value() == givenstring[0]:
            end = traverse(child, givenstring[1:])
            return end
    return False

def challenge(head, prefix, P1, P2):
    possible = traverse(head, prefix)
    if possible == False:
        return (P1, P2)
    if possible == True and len(possible) >= 4:
        return (P1, P2)
    return (P2, P1) 
    #1) do a traverse
    #2) if cannot advance (returned False), X
    #3) if at a word with no children AND 4 or more letters, X
    
    return (-1, -1)

def hint(head, prefix):
    possible = traverse(head, prefix)
    if possible == False or possible == True:
        return set()
    return possible
    #do a traverse for the word
    #if return False, then prefix does not work
    #otherwise, return the set

global digits
digits = "0123456789"
letters = "abcedfghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def main():
    start = time.time()
    if len(sys.argv) < 2:
        sys.argv.append(1)
    #OVERHEAD
    wordList = createList()
    #list happens to be sorted already
    headset = build(wordList)
    head = Node("", headset)
    print(time.time()-start)
    #currently can only have fewer than ten players 
    #GAME
    players = []
    lost = {}
    for i in range(0, sys.argv[1]):
        players.append(i+1)
        lost[i] = 0#count of loses
        
    pre = ""
    while True:
        getch = _Getch().__call__()
        if ord(getch) == 27:
            quit()
        else:
            player = players.pop()
            key = chr(ord(getch))
            sys.stdout.write(key)
            
            if key in digits:#challenge is only time player can lose
                winner, loser = challenge(head, pre, key, player)#returns the num of winner, key is challenger, player is the challengee
                sys.stdout.write("\nPlayer " + winner + " Won this Round")
                sys.stdout.write("\nPlayer " + loser + " Lost this Round")
                
                lost[loser] += 1#append lost count
                if lost[loser] == 5:
                    players.remove(loser)
                    sys.stdout.write("\nPlayer " + str(loser) + " is Out of the Game")
                
                #setTurn
                while players[0] != winner:
                    players.append(players.pop())
                    
                sys.stdout.write("\n")#new game
                
            elif key == ".":
                possible = hint(head, pre)
                sys.stdout.write("\n"+str(possible))
                if len(possible) == 0:
                    sys.stdout.write("Would you like to continue current game?  Hit ENTER if yes and BACKSPACE if no.")
                    while True:
                        answer = ord(_Getch().__call__())
                        if answer == 8: #backspace
                            sys.stdout.write("\n")
                            break
                        if answer == 13: #enter
                            sys.stdout.write("\n"+pre)
                            break
                    
            elif key not in letters:#if not a letter
                sys.stdout.write("\nINVALID KEY")
                sys.stdout.write("\n"+pre)
                
            else:#is a letter
                pre += key
                
            players.append(player)#put player back in queue

main()
