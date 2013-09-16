# Power digit sum
# Problem 16
# 215 = 32768 and the sum of its digits is 3 + 2 + 7 + 6 + 8 = 26.
# 
# What is the sum of the digits of the number 21000?

library(gmp)
sum(as.numeric(strsplit(as.character(pow.bigz(2,1000)[[1]]),'')[[1]]))
# 1366

# much easier in python, no additional modules necessary
# >>> sum([int(x) for x in str(2**1000)])
# 1366