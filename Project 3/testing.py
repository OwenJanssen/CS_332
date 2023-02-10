import nashpy as nash

A = [[3, 3], [2, 6]]
B = [[2, 7], [4, 8]]

game = nash.Game(A, B)
eq = game.support_enumeration()
for e in eq:
    print(e)