"""
    Traditional LSP game
    @author Mckay Kim
    @created 12-10-2016
    @teached by Mr.shin
    
"""

import random

# 0:gawi 1: bawi 2: bo
com_result = random.randint(0,2)

print "show ur hand(0:gawi 1: bawi 2: bo)"

my_hand = int (row_input());

a = my_hand - com_hand
if a>0 or a==-2:
        print "U win and I lose"
elif a==0:
        print "Draw"
else:
    print "U lose and I win"

    
"""
    Determine winner of the game/

"""
