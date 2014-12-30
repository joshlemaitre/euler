# problem 1
# sum(multiples of 3 < 1000 + multiples of 5 < 1000)
import time

def problem1():
	n = range(1,1000)
	numbers = []
	for number in n:
		if number % 3 == 0 or number % 5 == 0:
			numbers.append(number) 
	return(sum(numbers))

t0 = time.time()
ans = problem1()
t1 = time.time()
perf = t1 - t0

print "Problem 1\nAnswer:",ans, "runtime:", perf, "seconds"


# problem 2
# fibonacci numbers below 4e6
t0 = time.time()
l = [1,2]
while max(l) < 1e1001:
	l.append(l[len(l) - 2] + l[len(l) - 1])


l = l[:-1]
sums = []
for i in l:
	if i % 2 == 0:
		sums.append(i)

ans = sum(sums)
t1 = time.time()
perf = t1 - t0 
print "Problem 2\nAnswer:",ans, "runtime:", perf, "seconds"



# problem 3
# What is the largest prime factor of the number 600851475143?
def primes(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n /= d
        d += 1
    if n > 1:
       primfac.append(n)
    return primfac


t0 = time.time()
ans = max(primes(600851475143))
t1 = time.time()
perf = t1 - t0

print "Problem 3\nAnswer:",ans, "runtime:", perf, "seconds"



# problem 4
# Find the largest palindrome made from the product of two 3-digit numbers.
t0 = time.time()
n = range(100,1000)

def is_palindrome(n):
	return(str(n) == str(n)[::-1])

prods = []
for i in n:
	for j in n:
		prods.append(i*j)

pals = []
for prod in prods:
	if is_palindrome(prod) == True:
		pals.append(prod)

ans = max(pals)
t1 = time.time()
perf = t1 - t0

print "Problem 4\nAnswer:",ans, "runtime:", perf, "seconds"



# problem 5
# What is the smallest positive number that is evenly divisible 
# by all of the numbers from 1 to 20?
t0 = time.time()
import numpy

ints = numpy.array(range(1,21))
primes = [2,3,5,7,11,13,17,19] # under 20
facts = []
for p in primes:
	counter = 0
	nums = ints
	while any(nums % p == 0):
		nums = nums / float(p)
		counter += 1
	facts.append(counter)

facts = numpy.array(facts)
mults = primes**facts
ans = 1
for m in mults:
	ans = m * ans

t1 =time.time()
perf = t1 - t0
print "Problem 5\nAnswer:",ans, "runtime:", perf, "seconds"



# problem 6
# squared sum of 1-100 minus sum squares 1-100
t0 = time.time()
ans = sum(range(1,101))**2 - sum(map(lambda x: x**2, range(1,101)))
t1 = time.time()
perf = t1 - t0 
print "Problem 6\nAnswer:",ans, "runtime:", perf, "seconds"



# problem 7
# 10001th prime
t0 = time.time()
primes = [2,3,5,7,11,13,17,19]
n = range(23,1000000,2)
for i in n:
	if all(i % numpy.array(primes) != 0):# | len(primes) <= 10001):
		primes.append(i)

t1 = time.time()
perf = t1 - t0
ans = primes[10000]
print "Problem 7\nAnswer:",ans, "runtime:", perf, "seconds"



# problem 8
# 
t0 = time.time()
s = "7316717653133062491922511967442657474235534919493496983520312774506326239578318016984801869478851843858615607891129494954595017379583319528532088055111254069874715852386305071569329096329522744304355766896648950445244523161731856403098711121722383113622298934233803081353362766142828064444866452387493035890729629049156044077239071381051585930796086670172427121883998797908792274921901699720888093776657273330010533678812202354218097512545405947522435258490771167055601360483958644670632441572215539753697817977846174064955149290862569321978468622482839722413756570560574902614079729686524145351004748216637048440319989000889524345065854122758866688116427171479924442928230863465674813919123162824586178664583591245665294765456828489128831426076900422421902267105562632111110937054421750694165896040807198403850962455444362981230987879927244284909188845801561660979191338754992005240636899125607176060588611646710940507754100225698315520005593572972571636269561882670428252483600823257530420752963450"
prods = {}
for i in range(len(s)-13):
	n = s[i:i+13] 
	prods[n] = int(n[0]) * int(n[1]) * int(n[2]) * int(n[3]) * int(n[4]) * \
	int(n[5]) * int(n[6]) * int(n[7]) * int(n[8]) * int(n[9]) * int(n[10]) * \
	int(n[11]) * int(n[12])

ans = max(prods.values())
t1 = time.time()
perf = t1 - t0
print "Problem 8\nAnswer:",ans, "runtime:", perf, "seconds"



# problem 9
# pythag triple
# a < b < c & a + b + c = 1000 & a**2 + b**2 = c**2

t0 = time.time()
squares = {}
for i in range(1,1001):
	squares[i] = i**2

trips = {}
for key in squares.keys():
	for k in squares.keys():
		if int(key) < int(k) and 1000 - int(key) - int(k) > int(k) and squares[key] + squares[k] in squares.values():
			trips[key,k, (int(k)**2 + int(key)**2)**0.5] = int(k)+int(key)+(int(k)**2 + int(key)**2)**0.5
		else:
			pass

ans = {k: v for k, v in trips.iteritems() if v == 1000.0}
t1 = time.time()
perf = t1 - t0
print "Problem 9\nAnswer:",ans, "runtime:", perf, "seconds"



# problem 10
# sum of all primes below 2000000
t0 = time.time()
nums = set(range(2,2000000,1))
primes = []
while nums:
	p = nums.pop()
	primes.append(p)
	nums.difference_update(set(range(p*2,2000000,p)))

ans = sum(primes)

t1 = time.time()
perf = t1 - t0 
print "Problem 10\nAnswer:",ans,"runtime:", perf,"seconds"


# problem 11
# greatest product of 4 adjacent cells in a 20x20 matrix of numbers
# 08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08
# 49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00
# 81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65
# 52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91
# 22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80
# 24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50
# 32 98 81 28 64 23 67 10 26 38 40 67 59 54 70 66 18 38 64 70
# 67 26 20 68 02 62 12 20 95 63 94 39 63 08 40 91 66 49 94 21
# 24 55 58 05 66 73 99 26 97 17 78 78 96 83 14 88 34 89 63 72
# 21 36 23 09 75 00 76 44 20 45 35 14 00 61 33 97 34 31 33 95
# 78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92
# 16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57
# 86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58
# 19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40
# 04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66
# 88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69
# 04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36
# 20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16
# 20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54
# 01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48
# copy this to your clipboard via control c
# import pandas as pd
# df = pd.read_clipboard(header = None)
t0 = time.time()
df = np.matrix(df)
vmax = 0
hmax = 0 
rdmax = 0
ldmax = 0

for num in range(20):
    for start in range(16):
        n = reduce(lambda x,y: int(x*y), df[start:(start+4),num])
        if n > vmax:
            vmax = n

for num in range(20):
    for start in range(16):
        n = reduce(lambda x,y: int(x*y), df[num, start:(start+4)].T)
        if n > hmax:
            hmax = n

for num in range(16):
    for start in range(16):
    	n = df[num,start] * df[num+1,start+1] * df[num+2,start+2] * df[num+3,start+3]
    	if n > rdmax:
    		rdmax = n

for row in range(3,20):
    for col in range(16):
    	n = df[row,col] * df[row-1,col+1] * df[row-2,col+2] * df[row-3,col+3]
    	if n > rdmax:
    		rdmax = n

ans = max(vmax, hmax, rdmax, ldmax)
t1 = time.time()
perf = t1 - t0
print "Problem 11\nAnswer:",ans, "runtime:", perf, "seconds"


# problem 12
# triangle number (1,3,6,10,15,etc.) with 500 divisors
def num_divisors(n):
    if n % 2 == 0: n = n/2
    divisors = 1
    count = 0
    while n % 2 == 0:
        count += 1
        n = n/2
    divisors = divisors * (count + 1)
    p = 3
    while n != 1:
        count = 0
        while n % p == 0:
            count += 1
            n = n/p
        divisors = divisors * (count + 1)
        p += 2
    return divisors

 
def find_triangular_index(factor_limit):
    n = 1
    lnum, rnum = num_divisors(n), num_divisors(n+1)
    while lnum * rnum < 500:
        n += 1
        lnum, rnum = rnum, num_divisors(n+1)
    return n
 
start = time.time()
index = find_triangular_index(500)
triangle = (index * (index + 1)) / 2
elapsed = (time.time() - start)
 
print "result %s returned in %s seconds." % (triangle,elapsed)


# problem 13
# first 10 digits of the sum of these numbers:
# 37107287533902102798797998220837590246510135740250
# 46376937677490009712648124896970078050417018260538
# 74324986199524741059474233309513058123726617309629
# 91942213363574161572522430563301811072406154908250
# 23067588207539346171171980310421047513778063246676
# 89261670696623633820136378418383684178734361726757
# 28112879812849979408065481931592621691275889832738
# 44274228917432520321923589422876796487670272189318
# 47451445736001306439091167216856844588711603153276
# 70386486105843025439939619828917593665686757934951
# 62176457141856560629502157223196586755079324193331
# 64906352462741904929101432445813822663347944758178
# 92575867718337217661963751590579239728245598838407
# 58203565325359399008402633568948830189458628227828
# 80181199384826282014278194139940567587151170094390
# 35398664372827112653829987240784473053190104293586
# 86515506006295864861532075273371959191420517255829
# 71693888707715466499115593487603532921714970056938
# 54370070576826684624621495650076471787294438377604
# 53282654108756828443191190634694037855217779295145
# 36123272525000296071075082563815656710885258350721
# 45876576172410976447339110607218265236877223636045
# 17423706905851860660448207621209813287860733969412
# 81142660418086830619328460811191061556940512689692
# 51934325451728388641918047049293215058642563049483
# 62467221648435076201727918039944693004732956340691
# 15732444386908125794514089057706229429197107928209
# 55037687525678773091862540744969844508330393682126
# 18336384825330154686196124348767681297534375946515
# 80386287592878490201521685554828717201219257766954
# 78182833757993103614740356856449095527097864797581
# 16726320100436897842553539920931837441497806860984
# 48403098129077791799088218795327364475675590848030
# 87086987551392711854517078544161852424320693150332
# 59959406895756536782107074926966537676326235447210
# 69793950679652694742597709739166693763042633987085
# 41052684708299085211399427365734116182760315001271
# 65378607361501080857009149939512557028198746004375
# 35829035317434717326932123578154982629742552737307
# 94953759765105305946966067683156574377167401875275
# 88902802571733229619176668713819931811048770190271
# 25267680276078003013678680992525463401061632866526
# 36270218540497705585629946580636237993140746255962
# 24074486908231174977792365466257246923322810917141
# 91430288197103288597806669760892938638285025333403
# 34413065578016127815921815005561868836468420090470
# 23053081172816430487623791969842487255036638784583
# 11487696932154902810424020138335124462181441773470
# 63783299490636259666498587618221225225512486764533
# 67720186971698544312419572409913959008952310058822
# 95548255300263520781532296796249481641953868218774
# 76085327132285723110424803456124867697064507995236
# 37774242535411291684276865538926205024910326572967
# 23701913275725675285653248258265463092207058596522
# 29798860272258331913126375147341994889534765745501
# 18495701454879288984856827726077713721403798879715
# 38298203783031473527721580348144513491373226651381
# 34829543829199918180278916522431027392251122869539
# 40957953066405232632538044100059654939159879593635
# 29746152185502371307642255121183693803580388584903
# 41698116222072977186158236678424689157993532961922
# 62467957194401269043877107275048102390895523597457
# 23189706772547915061505504953922979530901129967519
# 86188088225875314529584099251203829009407770775672
# 11306739708304724483816533873502340845647058077308
# 82959174767140363198008187129011875491310547126581
# 97623331044818386269515456334926366572897563400500
# 42846280183517070527831839425882145521227251250327
# 55121603546981200581762165212827652751691296897789
# 32238195734329339946437501907836945765883352399886
# 75506164965184775180738168837861091527357929701337
# 62177842752192623401942399639168044983993173312731
# 32924185707147349566916674687634660915035914677504
# 99518671430235219628894890102423325116913619626622
# 73267460800591547471830798392868535206946944540724
# 76841822524674417161514036427982273348055556214818
# 97142617910342598647204516893989422179826088076852
# 87783646182799346313767754307809363333018982642090
# 10848802521674670883215120185883543223812876952786
# 71329612474782464538636993009049310363619763878039
# 62184073572399794223406235393808339651327408011116
# 66627891981488087797941876876144230030984490851411
# 60661826293682836764744779239180335110989069790714
# 85786944089552990653640447425576083659976645795096
# 66024396409905389607120198219976047599490197230297
# 64913982680032973156037120041377903785566085089252
# 16730939319872750275468906903707539413042652315011
# 94809377245048795150954100921645863754710598436791
# 78639167021187492431995700641917969777599028300699
# 15368713711936614952811305876380278410754449733078
# 40789923115535562561142322423255033685442488917353
# 44889911501440648020369068063960672322193204149535
# 41503128880339536053299340368006977710650566631954
# 81234880673210146739058568557934581403627822703280
# 82616570773948327592232845941706525094512325230608
# 22918802058777319719839450180888072429661980811197
# 77158542502016545090413245809786882778948721859617
# 72107838435069186155435662884062257473692284509516
# 20849603980134001723930671666823555245252804609722
# 53503534226472524250874054075591789781264330331690
t0 = time.time()
df = pd.read_clipboard(header = None)
df = list(df[0])
df = map(int, df)
ans = str(sum(df))[0:10]
perf = time.time() - t0


# problem 14
# collatz sequences under 1000000
t0 = time.time()
nums = range(1000000)
seqs = []
done = []
for number in nums:
	n = number
	idx = 0
	if n < number:
		idx = idx + seqs[n+1] 
	while n > 1:
		if n % 2 == 0:
			n  = n / 2
			idx = idx + 1
		else:
			n = 3*n + 1
			idx = idx + 1
	seqs.append(idx)
	done.append(number)

perf = time.time() - t0
ans = seqs.index(max(seqs))

print "result %s returned in %s seconds." % (ans,perf)


# problem 15
# how many down/right paths are there on a 20/20 grid
t0 = time.time()
import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

ans = nCr(40,20)
perf = time.time() - t0
print "result %s returned in %s seconds." % (ans,perf)


# problem 16
# 
t0 = time.time()
s = 0
for letter in str(2**1000):
     s = s + int(letter)

ans = s
perf = time.time() - t0
print "result %s returned in %s seconds." % (ans,perf)


# problem 17
# sum of letters in the numbers written from 1-1000
t0 = time.time()
tens = ['twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
nums = ['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
for ten in tens:
	nums.append(ten)

for ten in tens:
	for one in nums[:9]:
		nums.append(ten+one)

for one in nums[:9]:
	nums.append(one+'hundred')

for hundred in nums[:9]:
	for one in nums[:99]:
		nums.append(hundred+'hundredand'+one)

nums.append('onethousand')
ans = sum(map(len, nums))
perf = time.time() - t0
print "result %s returned in %s seconds." % (ans,perf)


# problem 18
# maximum sum to adjacent positions down triangle:
# 75
# 95 64
# 17 47 82
# 18 35 87 10
# 20 04 82 47 65
# 19 01 23 75 03 34
# 88 02 77 73 07 63 67
# 99 65 04 28 06 16 70 92
# 41 41 26 56 83 40 80 70 33
# 41 48 72 33 47 32 37 16 94 29
# 53 71 44 65 25 43 91 52 97 51 14
# 70 11 33 28 77 73 17 78 39 68 17 57
# 91 71 52 38 17 14 91 43 58 50 27 29 48
# 63 66 04 68 89 53 67 30 73 16 69 87 40 31
# 04 62 98 27 23 09 70 98 73 93 38 53 60 04 23
t0 = time.time()
rows = []
lines = []
f = open('/Users/u6022402/Documents/Personal/Euler/problem_18_data')
for line in f:
    rows.append([int(i) for i in line.rstrip('\n').split(" ")])

for i,j in [(i,j) for i in range(len(rows)-2,-1,-1) for j in range(i+1)]:
    rows[i][j] +=  max([rows[i+1][j],rows[i+1][j+1]])

ans = max(rows)
perf = time.time() - t0
print "result %s returned in %s seconds." % (ans,perf)



# problem 19
# How many Sundays fell on the first of the month during 
# the twentieth century (1 Jan 1901 to 31 Dec 2000)?
t0 = time.time()
import datetime
start = datetime.date(1901,1,1)
end = datetime.date(2000,12,31)
dates = pd.date_range(start, end)
firsts = []
for d in dates:
	if (d.weekday() == 6 and d.day == 1):
		firsts.append(d)

ans = len(firsts)
perf = time.time() - t0
print "result %s returned in %s seconds." % (ans,perf)



# problem 20
# sum of digits of 100!
t0 = time.time()
nums = range(1,101)
prod = 1
for num in nums:
    prod = prod * num

sums = 0 
for i in str(prod):
	sums = sums + int(i)

ans = sums
perf = time.time() - t0
print "result %s returned in %s seconds." % (ans,perf)


# problem 21
# sum of "amicable" number pairs below 10000
t0 = time.time()
def d(n):
	divs = [1]
	for num in range(1,(n+2)/2):
		if n % num == 0:
			divs.append(num)
	divs = list(set(divs))
	return sum(divs)

ds = []
nums = range(1,10001)
for num in nums:
	if num == d(d(num)) and num != d(num):
		ds.append(num)
		ds.append(d(num))

ans = sum(list(set(ds)))	
perf = time.time() - t0
print "result %s returned in %s seconds." % (ans,perf)


# problem 22
t0 = time.time()
nms = pd.read_csv("p022_names.txt")
nms = sorted(nms)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
alph = {}
for letter in alphabet:
    alph[letter] = alphabet.find(letter)+1

scores = {}
for name in nms:
	score = 0
	for letter in name:
		score += alph[letter]
		scores[name] = score

positions = {}
for name in nms:
	positions[name] = nms.index(name)+1

total = 0
for name in nms:
	total = total + positions[name]*scores[name]

ans = total
perf = time.time() - t0
print "result %s returned in %s seconds." % (ans,perf)


# problem 23
t0 = time.time()
def div_sum(n):
	divs = [1]
	for num in range(1,(n+2)/2):
		if n % num == 0:
			divs.append(num)
	divs = sorted(list(set(divs)))
	return sum(divs)

abun = []
for num in range(1,28124):
	if div_sum(num) > num:
		abun.append(num)

nums = range(1,28124)
s = set(nums)
for num in abun:
	for n2 in abun:
		s.difference_update([num + n2])

ans = sum(s)
t1 = time.time() - t0
print "result %s returned in %s seconds." % (ans,t1)

# L, s = 20162, 0
# abn = set()

# for n in range(1, L):
#     if div_sum(n) > n:
#         abn.add(n)
#     if not any( (n-a in abn) for a in abn ):
#         s += n



# problem 24
t0 = time.time()

from itertools import permutation, islice
def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)

ans = nth(permutations(range(10), 10), 1000000)
perf = time.time() - t0
print "result %s returned in %s seconds." % (ans,perf)


# problem 25
t0 = time.time()

l = [1,1,2]
while len(str(max(l))) < 1000:
	l.append(l[len(l) - 2] + l[len(l) - 1])
