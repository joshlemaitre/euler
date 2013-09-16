# Lattice paths
# Problem 15
# Starting in the top left corner of a 2x2 grid, there are 6 routes (without backtracking) to the bottom right corner.
# 
# 
# How many routes are there through a 20x20 grid?

# combinatorics problem: C(n, r) where n = 20+20 and r = 20
n <- 40; r <- 20
result <- choose(n,r)