# Largest prime factor
# Problem 3
# The prime factors of 13195 are 5, 7, 13 and 29.
#
# What is the largest prime factor of the number 600851475143 ?

euler003 <- function(y) {
x <- 1
z <- {}
orig <- y
while (x < y/2) {
    if (y %% x == 0)  {
      y <- y / x
      z <- c(unique(z), x)
      x <- x + 1
    } else { x <- x + 1 }
}
guess <- min(orig / cumprod(z)) 
if (sum(guess %% z == 0) > 1) {
    return(max(z)) } else { return(guess) }
}


euler003(600851475143)
