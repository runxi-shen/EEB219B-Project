""" EEB C219B Project Code
    The Effects of Reproductive Behaviors on the Incidence of
    Female Breast Cancer in China
"""
import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import random

# Colors for graphing
color = ['black', 'blue', 'brown', 'coral', 'cyan', 
         'darkgray','gray', 'green', 'lavender','darkred', 
         'burlywood', 'deeppink', 'beige', 'azure']
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
from copy import copy, deepcopy

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
def implement(tran, pr, pb):
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
                    tran[i][j] = round(pr[0] * pb[3], 4)
                elif (j == 4):
                    tran[i][j] = round(pr[0] * pb[0], 4)
            # from the states in the most right column, they can only goes down one by one
            # and the transitional probabilities will be the same as the breastfeeding duration time change probabilities
            if (i == 3 or i == 6 or i == 9):
                if j == i + 3:
                    (tran[i])[j] = round(pb[(i//3)-1], 4)
            # from all the intermediate states, there are three paths for change
            if (i == 1 or i == 4 or i == 7 or i == 2 or i == 5 or i == 8):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    (tran[i])[j] = round(pr[i%3], 4)
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    (tran[i])[j] = round(pb[i//3], 4)
                elif j == i + 4:
                    (tran[i])[j] = round(pr[i%3]*pb[i//3], 4)
            if (i == 10 or i == 11):
                if j == i + 1:
                    (tran[i])[j] = round(pr[i%3], 4)
                    
    # Implement the diagonal entries
    for k in range (len(tran)):
        s = 1 - sum(tran[k])
        (tran[k])[k] = round(s,4)

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
   of Chinese females in the past 30 years
   Second part of the implementation is the prediction of their breast cancer
   incident rate change based on their reproductive behaviors' change"""
   
"""First part:
   Reproductive behavior implementation in the past 30 years
"""
# *****************************************************************************
# Implementation for the 1980s
# -----------------------------------------------------------------------------
# Total Fertility rate data from the 1980s, for 1981, 1986, 1989
TFR1_80 = [1.184, 0.638, 0.792]
TFR2_80 = [1.113, 0.753, 0.499]
TFR3_80 = [1.043, 0.736, 0.516]

# List of the transitional probability vector for parous states each year in 1980~1989
pr80s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr80 = []

# Implementation for transitional probability vectors for parous states of each 
# year, due to the lackage of data, we will assume the vectors will be the same
# between 1980~1983 as the 1980's; vectors between 1984~!986 will be the same as
# 1986's; and vectors between 1987~1989 will be the same as 1989's

# By dividing each Total Fertility Rate by 30, (if we assume the fertility 
# cycle of a women will be from 20-50 years old) we can get an approximate 
# General Fertility Rate for each year per woman, which is the number of 
# children each woman will have in that year
for i in range (10):
    if (i <= 3):
        pr80.append(TFR1_80[0]/30)
        pr80.append(TFR1_80[1]/30)
        pr80.append(TFR1_80[2]/30)
    elif (3 < i < 7):
        pr80.append(TFR2_80[0]/30)
        pr80.append(TFR2_80[1]/30)
        pr80.append(TFR2_80[2]/30)
    else:
        pr80.append(TFR3_80[0]/30)
        pr80.append(TFR3_80[1]/30)
        pr80.append(TFR3_80[2]/30)
    pr80s.append(pr80)
    pr80 = []
# print (pr80s)
  
# Breastfeeding probability vector in 1980s
# Reference: "Association of menstrual and reproductive factors with breast 
# cancer risk: results from the Shanghai breast cancer study"
# Percentage of women who never breastfed
pb80_0 = 0.2090
# Percentage of women who breastfed less than 12 months
pb80_12 = 0.4274 
# Percentage of women who breastfed between 12 and 24 months
pb80_24 = 0.2021
# Percentage of women who breastfed more than 24 months
pb80_36 = 0.1615

#Average breastfeeding duration in 1980s = 12 months with variance 1.5
#pb80 = norm(loc=22, scale=20)
#pb80_12 = pb90.cdf(12)
#pb80_24 = pb90.cdf(24)-pb90.cdf(12)
#pb80_36 = pb90.cdf(36)-pb90.cdf(24)

# Transitional probabilities from different breastfeeding stages based on the
# percentages of women in each breastfeedin state
pb80s = [0.791, 0.4597, 0.4442, 0.2090]

# List of transitional matrices in 1990s
tran80s = []

# The empty transition matrix for each year, ready for implementation
tran80 = []

# Initialize the empty transitional matrix for 1980s
tran0_80 = deepcopy(tran0)

for i in range(len(pr80s)):
    tran80 = implement(tran0_80, pr80s[i], pb80s)
    # print (tran80)
    tran80s.append(tran80)
    tran0_80 = deepcopy(tran0)

# List of reproductive probabilities vectors for each year  
rpvs80 = []
# empty reproductive vector for implementation of each year
rpv80 = rs

for i in range(len(tran80s)):
    rpv80 = mul(rpv80, tran80s[i])
    rpvs80.append(rpv80)
print (rpvs80[len(rpvs80)-1])
    
fig80 = plt.gcf()
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
cgraph80 = []
for i in range (10):   
    g, = plt.plot(x, rpvs80[i], color=color[i])
    cgraph80.append(g)
fig80.legend([i for i in cgraph80], ['1980','1981','1982','1983','1984','1985',
            '1986','1987','1988','1989'], loc='center left', bbox_to_anchor=(0.65, 0.5))
plt.show()
# fig80.savefig('rs90s.pdf', dpi=100)
# -----------------------------------------------------------------------------

## Implementation for the 1990s
## -----------------------------------------------------------------------------
# Total Fertility rate data from the 1990s - 1990~1999, the number of children a woman has in her lifetime
TFR1_90 = [1.12,0.936,0.939,0.922,0.897,0.929,0.881,0.871,0.893,0.855]
TFR2_90 = [0.799,0.575,0.504,0.452,0.405,0.404,0.358,0.342,0.329,0.314]
TFR3_90 = [0.454,0.293,0.24,0.196,0.164,0.145,0.123,0.095,0.088,0.064]

# List of the transitional probability vector for parous states each year in 1990~1999
pr90s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr90 = []

# By dividing each Total Fertility Rate by 30, (if we assume the fertility 
# cycle of a women will be from 20-50 years old) we can get an approximate 
# General Fertility Rate for each year per woman, which is the number of 
# children each woman will have in that year
for i in range (10):
    pr90.append(TFR1_90[i]/30)
    pr90.append(TFR2_90[i]/30)
    pr90.append(TFR3_90[i]/30)
    pr90s.append(pr90)
    pr90 = []
# print (len(pr90s))
  
# Breastfeeding probability vector in 1990s
#Average breastfeeding duration in 1990s = 12 months with variance 1.5
#pb90 = norm(loc=22, scale=20)
#pb90_12 = pb90.cdf(12)
#pb90_24 = pb90.cdf(24)-pb90.cdf(12)
#pb90_36 = pb90.cdf(36)-pb90.cdf(24)

# imported from pb80s 
pb90s = [0.791, 0.4597, 0.4442, 0.2090]

# List of transitional matrices in 1990s
tran90s = []

# The empty transition matrix for each year, ready for implementation
tran90 = []

#print (pr90s)
tran0_90 = deepcopy(tran0)

for i in range(len(pr90s)):
    tran90 = implement(tran0_90, pr90s[i], pb90s)
    # print (tran90)
    tran90s.append(tran90)
    tran0_90 = deepcopy(tran0)
  
# List of reproductive probabilities vectors for each year  
rpvs90 = []
# empty reproductive vector for implementation of each year
rpv90 = rs

for i in range(len(tran90s)):
    rpv90 = mul(rpv90, tran90s[i])
    rpvs90.append(rpv90)
print (rpvs90[len(rpvs90)-1])

fig90 = plt.gcf()
cgraph90 = []
for i in range (10):   
    g, = plt.plot(x, rpvs90[i], color=color[i])
    cgraph90.append(g)
fig90.legend([i for i in cgraph90], ['1990','1991','1992','1993','1994','1995',
            '1996','1997','1998','1999'], loc='center left', bbox_to_anchor=(0.65, 0.5))
plt.show()
# fig90.savefig('rs90s.pdf', dpi=100)

# Implementation for the 2000s
# -----------------------------------------------------------------------------
# Fertility rate data from the 2000s
# General fertility rate 2003~2014, the number of children born among 1000 women
# Normalize the data by only focusing on women between 20-50 years old, and divide
# the total number of children they will have at each birth by the total sample size
# we get the general fertility rate for each year per woman
# The following data are in the order from 2014 to 2003
GFR1_00 = [0.0228,0.0251,0.0259	,0.0216,0.0232,0.0296,
           0.0327,0.0320,0.0307,0.0285,0.0340,0.0328]
GFR2_00 = [0.0152,0.0129,0.0130,0.0105,0.0126,0.0133,
           0.0137,0.0134,0.0129,0.0127,0.0117,0.0110]
GFR3_00 = [0.0028,0.0019,0.0019,0.0016,0.0026,0.0016,	
           0.0017,0.0016,0.0016,0.0021,0.0015,0.0020]
# Now we reverse the order of the data so make it go from 2003 to 2014 which 
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
for i in range (12):
    pr00.append(GFR1_00[i])
    pr00.append(GFR2_00[i])
    pr00.append(GFR3_00[i])
    pr00s.append(pr00)
    pr00 = []
# print (len(pr00s))
  
# Breastfeeding probability vector in 2000s
# Average breastfeeding duration in 2000s = 12 months with variance 1.5
pb00 = norm(loc=22, scale=20)
pb00_12 = pb00.cdf(12)
pb00_24 = pb00.cdf(24)-pb00.cdf(12)
pb00_36 = pb00.cdf(36)-pb00.cdf(24)

#print (pb00_12, pb00_24, pb00_36)
pb00s = [0.791, 0.4597, 0.4442, 0.2090]

# List of transitional matrices in 1990s
tran00s = []

# The empty transition matrix for each year, ready for implementation
tran00 = []

tran0_00 = deepcopy(tran0)
for i in range(len(pr00s)):
    tran00 = implement(tran0_00, pr00s[i], pb00s)
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
print (rpvs00[len(rpvs00)-11])

fig00 = plt.gcf()
cgraph00 = []
for i in range (11):
    g, = plt.plot(x, rpvs00[i], color=color[i])
    cgraph00.append(g)
plt.legend([i for i in cgraph00], ['2003','2004','2005','2006','2007','2008',
            '2009','2010','2011','2012','2013','2014'], 
            loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
# fig00.savefig('rs00s.pdf', dpi=100)

#g1, = plt.plot(x, rpvs80[len(rpvs80)-1], color=color[3])
#g2, = plt.plot(x, rpvs90[len(rpvs90)-1], color=color[1])
#g3, = plt.plot(x, rpvs00[len(rpvs00)-1], color=color[7])
#cgraph_total = [g1, g2, g3]
#plt.legend([i for i in cgraph_total], ['1980s','1990s','2000s'], 
#            loc='center left', bbox_to_anchor=(1, 0.5))
#plt.show()
# -----------------------------------------------------------------------------
# *****************************************************************************
# End of reproductive behavior simulation

"""Second part:
   Breast cancer incident rate change based on the change of reproductive
   behaviors' changes among Chinese females
"""























