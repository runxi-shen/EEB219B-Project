""" EEB C219B Project Code Version 2.0
    The Effects of Reproductive Behaviors on the Incidence of
    Female Breast Cancer in China
"""

import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt


# Colors for graphing
color = ['black', 'blue', 'brown', 'gold', 'cyan', 'darkgray',
         'gray', 'green', 'lavender','darkred','darkgoldenrod',
         'burlywood', 'deeppink', 'chocolate', 'indigo','darkorange',
         'lime']
# Problem setup
#------------------------------------------------------------------------------

# The probability vector for the 13 different reproductive states
# Every decade the vector will be reset to the initial value as below:
rs = [1,0,0,0,0,0,0,0,0,0,0,0,0]
# Women in childbearing age who haven't given birth yet

""" Corresponing reproductive states are
[0, 0],[1,  0],  [2,  0],  [3,  0]
       [1, <12], [2, <12], [3, <12]
       [1,12~24],[2,12~24],[3,12~24]
       [1, >24], [2, >24], [3, >24]
where the first entry represents the number of full term pregnancy,
and the second entry represents the breastfeeding duration.
"""

# Transition matrix setup
#------------------------------------------------------------------------------
from copy import deepcopy

# Initial transition matrix with all the entries to be zero
tran0 = []
tran00 = [0]*13
for i in range(len(tran00)):
    y = deepcopy(tran00)
    tran0.append(y)
#print (tran0)

""" 1. For each decade from 1980 to 2014, there are two vectors storing the
       transition probabilities for parous states and breastfeeding duration time
       states respectively.
    2. Vector pr corresponds to the probability of parous states transitions.
       pr = [p1, p2, p3], where p1 stands for the transitional probability of 
       a woman from having no child to having her first child, p2 stands for 
       the probability of her transiting from one-child state to two-child state,
       and p3 stands for the probability of her transiting from two-child state
       to three-child state.
    3. Vector pr corresponds to the probability of breastfeeding duration time
       states transitions.
       pb = [p1, p2, p3, p0], where p1 stands for the transitional probability of 
       a woman from never breastfeeding to breastfeeding for less than 12 months, 
       p2 stands for the probability of her transiting from breastfeeding for
       less than 12 months to breastfeeding for more than 12 months but less than
       24 months, and p3 stands for the probability of her transiting from 
       breastfeeding for more than 12 but less than 24 months to breastfeeding for
       more than 24 months p0 stands for the probability of her never breastfeeding
       in her lifetime
"""

# Implement the transition probability from states to states using the vectors
###############################################################################
def implement(tran, pr, pb1, pb2, pb3):
    for i in range(0, len(tran)):
        # In our setup, reproductive stages are sequential. 
        # So women can't go back to the stages they've been through, and can only
        # go to the states next to the states they stay in.
        # Here, implement the transition probabilities to other states first,
        # and the probability of staying at the same state will be the (1-sum(transitional probabilities))
        for j in range(i+1, len(tran)):
            # from [0,0] state, it can only go to [1,0]
            if (i == 0):
                if (j == 1):
                    tran[i][j] = pr[0] * pb1[3]
                elif (j == 4):
                    tran[i][j] = pr[0] * pb1[0]
            # from the states in the most right column, they can only goes down one by one
            # and the transitional probabilities will be the same as the breastfeeding duration time change probabilities
            if (i == 3 or i == 6 or i == 9):
                if j == i + 3:
                    (tran[i])[j] = pb3[(i//3)-1]
            # from all the intermediate states, there are three paths for change
            if (i == 1):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    (tran[i])[j] = pr[1]*pb2[3]
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    (tran[i])[j] = pb1[0]*(1-pr[1])
                elif j == i + 4:
                    (tran[i])[j] = pr[1]*pb2[0]
            if (i == 2):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    (tran[i])[j] = pr[2]*pb3[3]
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    (tran[i])[j] = pb2[0]*(1-pr[2])
                elif j == i + 4:
                    (tran[i])[j] = pr[2]*pb3[0]
            if (i == 4):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    (tran[i])[j] = pr[1]*(1-pb2[1])
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    (tran[i])[j] = pb1[1]*(1-pr[1])
                elif j == i + 4:
                    (tran[i])[j] = pr[1]*pb2[1]
            if (i == 5):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    (tran[i])[j] = pr[2]*(1-pb3[1])
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    (tran[i])[j] = pb2[1]*(1-pr[2])
                elif j == i + 4:
                    (tran[i])[j] = pr[2]*pb3[1]
            if (i == 7):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    (tran[i])[j] = pr[1]*(1-pb2[2])
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    (tran[i])[j] = pb1[3]*(1-pr[1])
                elif j == i + 4:
                    (tran[i])[j] = pr[1]*pb2[2]
            if (i == 8):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    (tran[i])[j] = pr[2]*(1-pb3[2])
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    (tran[i])[j] = pb2[2]*(1-pr[2])
                elif j == i + 4:
                    (tran[i])[j] = pr[2]*pb3[2]
            if (i == 10 or i == 11):
                if j == i + 1:
                    (tran[i])[j] = pr[i%3]
                    
    # Implement the diagonal entries
    for k in range (len(tran)):
        s = 1 - sum(tran[k])
        (tran[k])[k] = s

    return tran
###############################################################################

# Multiplication of row vector and the matrix
def mul(b, a):
    try:
        c = len(a[0])
        d = max(b)
    except:
        raise TypeError("The first entry should be vector, second should be matrix")
    bcopy = []
    sum = 0
    counter = 0
    b_index = 0
    a_index = 0
    while (counter < 13):
        for j in a:
            sum += b[b_index] * j[a_index]
            b_index += 1
        bcopy.append(round(sum, 4))
        sum = 0
        b_index = 0
        a_index += 1
        counter += 1
    return bcopy
#------------------------------------------------------------------------------
# End of the transition matrix setup
# End of Problem set-up

"""Implementations start here
   First part of implementation is the simulation of reproductive behaviors
   of Chinese females in their childbearing ages in the past fifteen years 
   (2000-2014)
   Second part of the implementation is the prediction of their breast cancer
   incident rate change based on their reproductive behaviors' change"""
   
"""First part:
   Reproductive behavior implementation in the past 15 years
"""
# *****************************************************************************

# Implementation for the 2000s 
# -----------------------------------------------------------------------------
# Fertility rate data from the 2000s (women start to bear children 20-24)
"""The model will simulate the reproductive behaviors of the cohort of 
 childbearing-age Chinese females in 2000-2014. Assuming that in 2000, a cohort
 of women in their childbearing age of 20~24 all start at initial reproductive states 
 ([1,0,0,0,0,0,0,0,0,0,0,0,0]), we can then use data for general fertility rate
 (GFR) in different age groups of females, 20~24, 25~29, 30~35 to simulate their 
 reproductive behaviors in fifteen years. For instance, from 2000~2004, the average 
 GFR of the first, second and thrid birthe for women aged from 20~24 will be 
 implemented to calculate the transitional probabilities of women in different
 parous states; and similarly, in 2005~2009, the data for women aged 25~29 will 
 be used and in 2010~2014, the data for women aged 30~34 will be used. 
 The following data are in the order from 2014 to 2000 (the data for 2001 and 
 2002 are missing, use the data from 2000 to estimate 2001 and data from 2003 
 to estimate 2002)
"""
# Breastfeeding probability vector in 2000s
# Use the data from "The changes in female physical and childbearing
# characteristics in china and potential association with risk of breast cancer"
# Based on the research in 2012, for the cohort of women aged 25-34, the average
# number of parity they have is 1.01, and their accumulative breastfeeding period
# has a mean of 13.62 and SD 9.80. We can draw a normal distribution from the data
# to estimate the breastfeeding duration time per child for women in 2000s.
#fig = plt.figure()
mu1, sigma1 = 13.62, 9.80
pb01 = norm(loc=mu1,scale=sigma1)
#mean, var, skew, kurt = norm.stats(loc=13.62,scale=sigma1,moments='mvsk')
#print (mean,var)
""" For every child a woman gives birth to, the probability of her devoting no 
    breastfeeding is pb01_0, probability of her devoting 0~12 months breastfeeding
    is pb01_12, and 12~24, more than 24 months breastfeeding are pb01_24, pb01_36. 
"""
pb01_0 = pb01.cdf(0)
pb01_12 = pb01.cdf(12)-pb01.cdf(0)
pb01_24 = pb01.cdf(24)-pb01.cdf(12)
pb01_36 = 1-pb01.cdf(24)
#x = np.linspace(-20, 60, 1000)
#plt.plot(x, pb01.pdf(x))
#fig.set_size_inches(18.5,10.5)
#print (pb01_12, pb01_24, pb01_36, pb01_0)
#pb00s = [pb01_12/(1-pb01_0),pb01_24/(1-pb01_0),pb01_36/(1-pb01_0-pb01_12),pb01_0]
pb00s_1 = [pb01_12/(1-pb01_0),pb01_24/(1-pb01_0),pb01_36/(1-pb01_0-pb01_12),pb01_0]
#print (pb00s_1)

""" If a woman bears two children, the probability of her devoting no 
    breastfeeding is pb02_0, probability of her devoting 0~12 months breastfeeding
    is pb02_12, and 12~24, more than 24 months breastfeeding are pb02_24, pb02_36. 
"""
pb02 = np.random.normal(mu1,sigma1,1000)
pb02_ = np.random.normal(mu1,sigma1,1000)
pb02s = pb02 + pb02_
(mu2, sigma2) = norm.fit(pb02s)
#print (mu2, sigma2)
pb02 = norm(loc=mu2,scale=sigma2)
#n, bins, patches = plt.hist(pb02s, 60, normed=1, facecolor='green', alpha=0.75)
#
## add a 'best fit' line
#y = matplotlib.mlab.normpdf( bins, mu2, sigma2)
#l = plt.plot(bins, y, 'r--', linewidth=2)
#mean, var, skew, kurt = norm.stats(loc=13.62,scale=sigma1,moments='mvsk')
#print (mean,var)
pb02_0 = pb02.cdf(0)
pb02_12 = pb02.cdf(12)-pb02.cdf(0)
pb02_24 = pb02.cdf(24)-pb02.cdf(12)
pb02_36 = 1-pb02.cdf(24)
#x = np.linspace(0, 50, 1000)
#plt.plot(x, pb02.pdf(x))
#print (pb02_12, pb02_24, pb02_36, pb02_0)
pb00s_2 = [pb02_12/(1-pb02_0),pb02_24/(1-pb02_0),pb02_36/(1-pb02_0-pb02_12),pb02_0]
#print (pb00s_2)

""" If a woman bears three children, the probability of her devoting no 
    breastfeeding is pb03_0, probability of her devoting 0~12 months breastfeeding
    is pb03_12, and 12~24, more than 24 months breastfeeding are pb03_24, pb03_36. 
"""
pb03_ = np.random.normal(mu1,sigma1,1000)
pb03s = pb02s + pb03_
(mu3,sigma3) = norm.fit(pb03s)
#print (mu3,sigma3)
pb03 = norm(loc=mu3,scale=sigma3)
#y = matplotlib.mlab.normpdf( bins, mu3, sigma3)
#l = plt.plot(bins, y, 'r--', color=color[3],linewidth=2)
pb03_0 = pb03.cdf(0)
pb03_12 = pb03.cdf(12)-pb03.cdf(0)
pb03_24 = pb03.cdf(24)-pb03.cdf(12)
pb03_36 = 1-pb03.cdf(24)
#print (pb03_12, pb03_24, pb03_36, pb03_0)
pb00s_3 = [pb03_12/(1-pb03_0),pb03_24/(1-pb03_0),pb03_36/(1-pb03_0-pb03_12),pb03_0]
#print (pb00s_3)
# pb00s = [0.791, 0.4597, 0.4442, 0.2090] // from other data sources


# Fertility rate data from the 2000s (women start to bear children 20-24)
GFR1_00 = [0.01701,0.02284,0.02221,0.01545,0.0174,
           0.06457,0.07008,0.07353,0.06875,0.05688,
           0.11169,0.11173,0.11173,0.10262,0.10262]
GFR2_00 = [0.02574,0.02415,0.02446,0.02177,0.02344,
           0.02933,0.02825,0.0279,0.02777,0.0304,
           0.0081,0.00904,0.00904,0.0104,0.0104]
GFR3_00 = [0.00625,0.00376,0.00405,0.00363,0.005,
           0.00268,0.00311,0.00235,0.00318,0.00442,
           0.00106,0.0019,0.0019,0.00146,0.00146]
# Now we reverse the order of the data so make it go from 2000 to 2014 which 
# will be easier to implement
GFR1_00.reverse()
GFR2_00.reverse()
GFR3_00.reverse()

# List of the transitional probability vector for parous states each year in 1990~1999
pr00s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr00 = []
# By dividing each Total Fertility Rate by 30, (if we assume the fertility cycle of a women
# will be from 20-50 years old) we can get an approximate General Fertility
# Rate for each year per woman, which is the number of children each woman will have in that year
for i in range (15):
    pr00.append(GFR1_00[i])
    pr00.append(GFR2_00[i])
    pr00.append(GFR3_00[i])
    pr00s.append(pr00)
    pr00 = []
# print (len(pr00s))

# List of transitional matrices in 1990s
tran00s = []

# The empty transition matrix for each year, ready for implementation
tran00 = []

tran0_00 = deepcopy(tran0)
for i in range(len(pr00s)):
    tran00 = implement(tran0_00, pr00s[i], pb00s_1, pb00s_2, pb00s_3)
    tran00s.append(tran00)
    tran0_00 = deepcopy(tran0)
# print (tran00s)
    
# List of reproductive probabilities vectors for each year  
rpvs00 = []
# empty reproductive vector for implementation of each year
rpv00 = rs

for i in range(len(tran00s)):
    rpv00 = mul(rpv00, tran00s[i])
    rpvs00.append(rpv00)
#    print (rpv00)

fig00_20 = plt.gcf()
fig00_20.set_size_inches(18.5,10.5)
x = [x for x in range (1,14)];
cgraph00 = []
for i in range (15):
    g, = plt.plot(x, rpvs00[i], color=color[i])
    cgraph00.append(g)
fig00_20.suptitle('Reproductive Vectors for 20-24 group', fontsize = 24)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in each state', fontsize = 24)
plt.legend([i for i in cgraph00], ['2000','2001','2002','2003','2004','2005',
            '2006','2007','2008','2009','2010','2011','2012','2013','2014'], 
            loc='lower left', bbox_to_anchor=(1, 0.5))
plt.show()
#fig00.savefig('rs00s.pdf', dpi=100)
#fig00_20.savefig('rs00s.png', dpi=100)

# (women start to bear children 25-29)
"""The model will simulate the reproductive behaviors of the cohort of 
 childbearing-age Chinese females in 2000-2014. Assuming that in 2000, a cohort
 of women in their childbearing age of 25~29 all start at initial reproductive states 
 ([1,0,0,0,0,0,0,0,0,0,0,0,0]), we can then use data for general fertility rate
 (GFR) in different age groups of females, 25~29, 30~34, 35~39 to simulate their 
 reproductive behaviors in fifteen years. For instance, from 2000~2004, the average 
 GFR of the first, second and thrid birthe for women aged from 25~29 will be 
 implemented to calculate the transitional probabilities of women in different
 parous states; and similarly, in 2005~2009, the data for women aged 30~34 will 
 be used and in 2010~2014, the data for women aged 35~39 will be used. 
 The following data are in the order from 2014 to 2000 (the data for 2001 and 
 2002 are missing, use the data from 2000 to estimate 2001 and data from 2003 
 to estimate 2002)
"""
GFR1_25 = [0.0043,0.00596,0.0048,0.0031,0.00548,
           0.0196,0.02216,0.01965,0.01485,0.00912,
           0.07595,0.07086,0.07086,0.05544,0.05544]
GFR2_25 = [0.00971,0.00995,0.00991,0.00766,0.01011,
           0.02762,0.02803,0.02853,0.02923,0.02689,
           0.02855,0.02768,0.02768,0.02581,0.02581]
GFR3_25 = [0.00301,0.00268,0.00244,0.00179,0.00312,
           0.00268,0.00311,0.00235,0.00318,0.00442,
           0.0031,0.00391,0.00391,0.00494,0.00494]
# Now we reverse the order of the data so make it go from 2000 to 2014 which 
# will be easier to implement
GFR1_25.reverse()
GFR2_25.reverse()
GFR3_25.reverse()

# List of the transitional probability vector for parous states each year in 1990~1999
pr25s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr25 = []
# By dividing each Total Fertility Rate by 30, (if we assume the fertility cycle of a women
# will be from 20-50 years old) we can get an approximate General Fertility
# Rate for each year per woman, which is the number of children each woman will have in that year
for i in range (15):
    pr25.append(GFR1_25[i])
    pr25.append(GFR2_25[i])
    pr25.append(GFR3_25[i])
    pr25s.append(pr25)
    pr25 = []
# print (len(pr00s))

# List of transitional matrices in 1990s
tran25s = []

# The empty transition matrix for each year, ready for implementation
tran25 = []

tran0_25 = deepcopy(tran0)
for i in range(len(pr25s)):
    tran25 = implement(tran0_25, pr25s[i], pb00s_1, pb00s_2, pb00s_3)
    tran25s.append(tran25)
    tran0_25 = deepcopy(tran0)
# print (tran25s)
    
# List of reproductive probabilities vectors for each year  
rpvs25 = []
# empty reproductive vector for implementation of each year
rpv25 = rs

for i in range(len(tran25s)):
    rpv25 = mul(rpv25, tran25s[i])
    rpvs25.append(rpv25)
#   print (rpv25)

fig00_25 = plt.gcf()
fig00_25.set_size_inches(18.5,10.5)
x = [x for x in range (1,14)];
cgraph00 = []
for i in range (15):
    g, = plt.plot(x, rpvs25[i], color=color[i])
    cgraph00.append(g)
fig00_25.suptitle('Reproductive Vectors for 25-29 group', fontsize = 24)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in each state', fontsize = 24)
plt.legend([i for i in cgraph00], ['2000','2001','2002','2003','2004','2005',
            '2006','2007','2008','2009','2010','2011','2012','2013','2014'], 
            loc='lower left', bbox_to_anchor=(1, 0.5))
plt.show()
##fig25.savefig('rs00s.pdf', dpi=100)
#fig00_25.savefig('rs00s.png', dpi=100)

# -----------------------------------------------------------------------------
# *****************************************************************************
# End of reproductive behavior simulation

"""Second part:
   Breast cancer incident rate change based on the change of reproductive
   behaviors' changes among Chinese females
"""
# -----------------------------------------------------------------------------
"""For this part, we apply the idea of "relative risk index" to calculate the 
   increasing incidence rate of breast cancer among the population. Each 
   reproductive will have a corresponding relative risk index to represent the 
   chance of women in that states developing a breast cancer in her lifetime.
   Relative risk of breast cancer is reduced by 4.3% (95% CI 2.9-5.8) for each
   year that a woman breastfeeds, in addition to a reduction of 7.0% (5.9-9.0)
   for each birth.
"""
## Implementation of index vector for the thirteen different reproductive stages
## The trend in the relative risk of breast cancer with increasing duration of
## breastfeeding is calculated. In such instances, the duration of breastfeeding
## associated with a particular category is taken to be the median duration within
## that category.
## Assumption: women with no child who had never breastfed are taken to have a 
## relative risk of 1.0.
risk_index = [1.0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range (13):
    if (0 < i < 4):
        risk_index[i] = risk_index[i-1]*(1-0.07)
    elif (3 < i < 7):
        risk_index[i] = risk_index[i-3]*(1-0.0215)
    elif (6 < i < 10):
        risk_index[i] = risk_index[i-3]*(1-0.043)
    elif (9 < i < 13):
        risk_index[i] = risk_index[i-3]*(1-0.043)
#print (risk_index) 
""" For each year in 2000s, based on the women's reproductive state, we are
    gonna use the relative risk index to calculate their relative risk of 
    developing breast cancer
"""
sum00 = []
sum1 = 0
for i in rpvs00:
    for j in range (13):
        sum1 += i[j]*risk_index[j]
    sum00.append(sum1)
    sum1 = 0
#print (sum00)
figrisk20 = plt.gcf()
figrisk20.set_size_inches(18.5,10.5)
x = [x for x in range (2000,2015)];
plt.plot(x,sum00)
plt.xticks(np.arange(min(x), max(x)+1, 1))
plt.xlabel('Time/year', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
##figrisk29.savefig('risk_index20.pdf', dpi=100)
#figrisk20.savefig('risk_index20.png', dpi=100)

sum25 = []
sum1 = 0
for i in rpvs25:
    for j in range (13):
        sum1 += i[j]*risk_index[j]
    sum25.append(sum1)
    sum1 = 0
#print (sum25)
figrisk25 = plt.gcf()
figrisk25.set_size_inches(18.5,10.5)
x = [x for x in range (2000,2015)];
plt.plot(x,sum00)
plt.xticks(np.arange(min(x), max(x)+1, 1))
plt.xlabel('Time/year', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
##figrisk25.savefig('risk_index25.pdf', dpi=100)
#figrisk25.savefig('risk_index25.png', dpi=100)
# -----------------------------------------------------------------------------
























