# Largest palindrome product
# Problem 4
# A palindromic number reads the same both ways. The largest palindrome made from the product of two 2-digit numbers is 9009 = 91 99.

# Find the largest palindrome made from the product of two 3-digit numbers.

isPalindrome <- function(x) {
    x <- as.character(x)
    rx <- strsplit(x,'')[[1]]
    rx <- paste(rev(rx),collapse = '')
    rx == x
}

prods <- merge(100:999,100:999,all = TRUE)
prods$prod <- prods[,1] * prods[,2]
prods$pal <- sapply(prods$prod, isPalindrome)
max(prods$prod[prods$pal])

