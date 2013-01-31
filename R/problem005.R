# Smallest multiple
# Problem 5
# 2520 is the smallest number that can be divided by each of the numbers from 1 to 10 without any remainder.

# What is the smallest positive number that is evenly divisible by all of the numbers from 1 to 20?

fac <- function(x) {
    y <- 2
    f <- 1
    while (y <= x ) {
    if (x %% y == 0) {
        f <- c(f, y)
        x <- x / y } else {
        y <- y + 1 }
    }
    return(f)
}

facs <- unlist(sapply(1:20, FUN = function(x) table(fac(x))))
agg <- aggregate(facs ~ as.numeric(names(facs)), FUN = max)
max(cumprod(agg[,1] ^ agg[,2]))