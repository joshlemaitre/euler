# Special Pythagorean triplet
# Problem 9
# A Pythagorean triplet is a set of three natural numbers, a  b  c, for which,

# a^2 + b^2 = c^2
# For example, 3^2 + 4^2 = 9 + 16 = 25 = 5^2.

# There exists exactly one Pythagorean triplet for which a + b + c = 1000.
# Find the product abc.

a <- 1:333
b <- 333:1000

for (i in seq_along(a)) {
    for (j in seq_along(b)) {
        test <- a[i] + b[j] + sqrt(a[i]^2 + b[j]^2)
        if (test == 1000) {
            win.a <- a[i]
            win.b <- b[j]
            break
        }
    }
}

win.a*win.b*sqrt(win.a^2 + win.b^2)