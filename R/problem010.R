# Summation of primes
# Problem 10
# The sum of the primes below 10 is 2 + 3 + 5 + 7 = 17.

# Find the sum of all the primes below two million.
m <- 2000000
primes <- c(2,3)
n <- 5
while (max(primes^2) < m) {
if(any(n %% primes[primes < n/2] == 0)) n <- n + 2 
else { primes <- c(primes,n)
    n <- n + 2 }
}

primes <- primes[-length(primes)]

candidates <- 2:m
for (i in 1:10) {
  candidates <- candidates[candidates %% primes[i] != 0 | candidates %in% primes]
}

prime <- function(x) {
  s <- sqrt(x)
  if(all(x %% primes != 0) | x %in% primes) {return(1)}
  else {return(0)}
}

flags <- sapply(candidates, prime)
sum(as.numeric(candidates[flags == 1]), na.rm = T)
