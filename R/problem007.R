# 10001st prime
# Problem 7
# By listing the first six prime numbers: 2, 3, 5, 7, 11, and 13, we can see that the 6th prime is 13.

# What is the 10 001st prime number?


n <- 10001
prime.seq <- numeric(n)
prime.seq[1:2] <- c(2, 3)
candidate <- prime.seq[2] + 2
for (i in 3: n) {
  while (any(candidate %% prime.seq[1:(i - 1)] == 0)) {
    candidate <- candidate + 2
  }
  prime.seq[i] <- candidate
  candidate <- candidate + 2
}

prime.seq[n]