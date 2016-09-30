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
""" For every child a woman gives birth to, the probability of her devoting no 
    breastfeeding is pb01_0, probability of her devoting 0~12 months breastfeeding
    is pb01_12, and 12~24, more than 24 months breastfeeding are pb01_24, pb01_36. 
"""
pb01_0 = pb01.cdf(0)
pb01_12 = pb01.cdf(12)-pb01.cdf(0)
pb01_24 = pb01.cdf(24)-pb01.cdf(12)
pb01_36 = 1-pb01.cdf(24)
pb00s_1 = [pb01_12/(1-pb01_0),pb01_24/(1-pb01_0),pb01_36/(1-pb01_0-pb01_12),pb01_0]

""" If a woman bears two children, the probability of her devoting no 
    breastfeeding is pb02_0, probability of her devoting 0~12 months breastfeeding
    is pb02_12, and 12~24, more than 24 months breastfeeding are pb02_24, pb02_36. 
"""
pb02 = np.random.normal(mu1,sigma1,1000)
pb02_ = np.random.normal(mu1,sigma1,1000)
pb02s = pb02 + pb02_
(mu2, sigma2) = norm.fit(pb02s)
pb02 = norm(loc=mu2,scale=sigma2)
pb02_0 = pb02.cdf(0)
pb02_12 = pb02.cdf(12)-pb02.cdf(0)
pb02_24 = pb02.cdf(24)-pb02.cdf(12)
pb02_36 = 1-pb02.cdf(24)
pb00s_2 = [pb02_12/(1-pb02_0),pb02_24/(1-pb02_0),pb02_36/(1-pb02_0-pb02_12),pb02_0]

""" If a woman bears three children, the probability of her devoting no 
    breastfeeding is pb03_0, probability of her devoting 0~12 months breastfeeding
    is pb03_12, and 12~24, more than 24 months breastfeeding are pb03_24, pb03_36. 
"""
pb03_ = np.random.normal(mu1,sigma1,1000)
pb03s = pb02s + pb03_
(mu3,sigma3) = norm.fit(pb03s)
pb03 = norm(loc=mu3,scale=sigma3)
pb03_0 = pb03.cdf(0)
pb03_12 = pb03.cdf(12)-pb03.cdf(0)
pb03_24 = pb03.cdf(24)-pb03.cdf(12)
pb03_36 = 1-pb03.cdf(24)
pb00s_3 = [pb03_12/(1-pb03_0),pb03_24/(1-pb03_0),pb03_36/(1-pb03_0-pb03_12),pb03_0]

# Fertility rate data from the 2000s
"""The model will simulate the reproductive behaviors of different cohorts of 
 childbearing-age Chinese females in 2000-2014. Assuming that in 2000, women
 at different ages all start haven't born any child and all at our initial 
 reproductive states([1,0,0,0,0,0,0,0,0,0,0,0,0]), we can then use data for 
 general fertility rate (GFR)  of women in different ages to simulate their 
 reproductive behaviors in fifteen years. For instance, for a women start to 
 her reproductive cycle at her 20 in 2000, every year we will update the GFR data 
 by the change of her age, in 2001 we will use the data for 21-year-old women
 (the data for 2001 and 2002 are missing, use the data from 2000 to estimate 
 2001 and data from 2003 to estimate 2002). I will use the model to simulate 
 different reproductive patterns if women start to bear children from at their 
 20 to at their 35 and see how the delay of the age to start childbearing process
 affects their parous and breastfeeding states. Assume that all the women will 
 end their reproductive cycle at 44. For the years after 2014, we will use the 
 data from 2014 to estimate their GFR and transitional probabilities.
"""

""" A list for final reproductive vectors of women who start to bear children
    at different ages"""    
frpv = []

""" Fertility rate data in the 2000s (women start to bear children at 20) """
GFR1_20 = [52.56, 90.82, 128.89, 137.67, 134.96, 93.96, 85.87, 77.87, 59.28, 44.81, 
           27.65, 20.40, 19.69, 17.79, 9.34, 8.16, 5, 4.18, 2.83, 1.6,
           0.94, 1.4, 0.81, 0.57, 0.09]
GFR2_20 = [0.00, 6.04, 7.99, 12.82, 14.17, 22.68, 26.39, 31.63,
           33.54, 32.35, 26.72, 24.53, 24.14, 22.30, 20.29, 16.57,
           11.09, 10.13, 5.77, 5.72, 3.99, 2.21, 2.94, 1.14, 1.04]
GFR3_20 = [0.00, 0.00, 1.51, 2.87, 2.15, 3.05, 3.53, 2.76, 3.73,
           4.6, 5.42, 4.58, 4.97, 4.99, 5.35, 5.02, 2.84, 2.66, 2.95,
           1.72, 1.05, 1.3, 1.12, 0.76, 0.57]

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr20s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr20 = []

for i in range (len(GFR1_20)):
#    sum1, sum2, sum3 = 1, 1, 1   
    product1, product2, product3 = 1, 1, 1
    if (i == 0):
        pr20.append(GFR1_20[i]/1000)
        pr20.append(GFR2_20[i]/1000)
        pr20.append(GFR3_20[i]/1000)
    else:
#        for j in range (i):
#            sum1 += GFR1_20[j]
#            sum2 += GFR2_20[j]
#            sum3 += GFR3_20[j]
#        pr20.append(GFR1_20[i]/(1000-sum1))
#        pr20.append(GFR2_20[i]/(1000-sum2))
#        pr20.append(GFR3_20[i]/(1000-sum3))
    
#        product1, product2, product3 = 1, 1, 1
#        for j in range (i):
#            product1 *= (1-pr20s[j][0])
#            product2 *= (1-pr20s[j][1])
#            product3 *= (1-pr20s[j][2])
#        pr20.append(GFR1_20[i]/1000/product1)
#        pr20.appe(ndGFR2_20[i]/1000/product2)
#        pr20.append(GFR3_20[i]/1000/product3)
        
#        product1 = (1-pr20s[i-1][0])
#        product2 = (1-pr20s[i-1][1])
#        product3 = (1-pr20s[i-1][2])
#        pr20.append(GFR1_20[i]/1000/product1)
#        pr20.append(GFR2_20[i]/1000/product2)
#        pr20.append(GFR3_20[i]/1000/product3)
    
        product1, product2, product3 = 1, 1, 1
        for j in range (i):
            product1 *= (1-GFR1_20[j]/1000)
            product2 *= (1-GFR2_20[j]/1000)
            product3 *= (1-GFR3_20[j]/1000)
        pr20.append(GFR1_20[i]/1000/product1)
        pr20.append(GFR2_20[i]/1000/product2)
        pr20.append(GFR3_20[i]/1000/product3)
    pr20s.append(pr20)
#    print (product1, product2, product3)
#    print (i, pr20, '\n')
    pr20 = []
#print (pr20s)

# List of transitional matrices in 2000s for 20
tran20s = []

# The empty transition matrix for each year, ready for implementation
tran20 = []

tran20 = deepcopy(tran0)
for i in range(len(pr20s)):
    tran20 = implement(tran20, pr20s[i], pb00s_1, pb00s_2, pb00s_3)    
    tran20s.append(tran20)
    tran20 = deepcopy(tran0)
# print (tran00s)
print (tran20s[0])

## List of reproductive probabilities vectors for each year  
#rpvs20 = []
## empty reproductive vector for implementation of each year
#rpv20 = rs
#
#for i in range(len(tran20s)):
#    rpv20 = mul(rpv20, tran20s[i])
##   rpvs20.append(rpv20)
##   print (rpv00)
#frpv.append(rpv20)
#
#""" Fertility rate data in the 2000s (women start to bear children at 21) """
#
##GFR1_21 = [90.82, 128.89, 137.67, 134.96, 93.96, 85.87, 
##           77.87, 59.28, 44.81, 27.65, 20.40, 19.69, 17.79, 9.34, 
##           8.16, 5, 4.18, 2.83, 1.6, 0.94, 1.4, 0.81, 0.57, 0.09]
##GFR2_21 = [0.00, 7.99, 12.82, 14.17, 22.68, 26.39, 31.63,
##           33.54, 32.35, 26.72, 24.53, 24.14, 22.30, 20.29, 16.57,
##           11.09, 10.13, 5.77, 5.72, 3.99, 2.21, 2.94, 1.14, 1.04]
##GFR3_21 = [0.00, 0.00, 2.87, 2.15, 3.05, 3.53, 2.76, 3.73,
##           4.6, 5.42, 4.58, 4.97, 4.99, 5.35, 5.02, 2.84, 2.66, 2.95,
##           1.72, 1.05, 1.3, 1.12, 0.76, 0.57]
#
## List of the transitional probability vector for parous states each year in 
## 2000-2014, for women start to bear children at 21 in 2000
#pr21s = deepcopy(pr20s[1:])
#pr21s[0][1] = 0.0
#pr21s[1][2] = 0.0
## Each year's transitional probability vector for parous states, ready for implementation
##for i in pr21s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 21
#tran21s = []
#
## The empty transition matrix for each year, ready for implementation
#tran21 = []
#
#tran21 = deepcopy(tran0)
#for i in range(len(pr21s)):
#    tran21 = implement(tran21, pr21s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran21s.append(tran21)
#    tran21 = deepcopy(tran0)
## print (tran21s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs21 = []
## empty reproductive vector for implementation of each year
#rpv21 = rs
#
#for i in range(len(tran21s)):
#    rpv21 = mul(rpv21, tran21s[i])
##   rpvs21.append(rpv21)
##   print (rpv21)
#frpv.append(rpv21)
#
#""" Fertility rate data in the 2000s (women start to bear children at 22) """
#
##GFR1_22 = [114.22, 126.66, 137.86, 121.19, 102.24, 55.31, 48.66,
##           44.29, 34.14, 24.05, 17.04, 9.93, 11.35, 10.35, 5.00]
##GFR2_22 = [0.00, 14.00, 16.04, 20.54, 25.23, 31.07, 33.05, 33.61,
##           33.26, 29.22, 25.92, 21.83, 16.20, 15.64, 11.09]
##GFR3_22 = [0.00, 0.00, 3.49, 3.45, 2.89, 4.41, 4.45, 1.55, 4.93,
##           4.09, 5.46, 3.34, 4.32, 3.36, 2.84]
#        
## List of the transitional probability vector for parous states each year in 
## 2000-2014, for women start to bear children at 22 in 2000
#pr22s = []
## Each year's transitional probability vector for parous states, ready for implementation
#pr22 = []
#
##for i in range (15):
##    if (i == 0):
##        pr22.append(GFR1_22[i]/1000)
##        pr22.append(GFR2_22[i]/1000)
##        pr22.append(GFR3_22[i]/1000)
##    else:
##        product1, product2, product3 = 1, 1, 1
##        for j in range (i):
##            product1 *= (1 - GFR1_22[j]/1000)
##            product2 *= (1 - GFR2_22[j]/1000)
##            product3 *= (1 - GFR3_22[j]/1000)
##        pr22.append((GFR1_22[i]/1000)/product1)
##        pr22.append((GFR2_22[i]/1000)/product2)
##        pr22.append((GFR3_22[i]/1000)/product3)
##    pr22s.append(pr22)
##    pr22 = []
#pr22s = deepcopy(pr21s[1:])
#pr22s[0][1] = 0.0
#pr22s[1][2] = 0.0
##for i in pr22s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 22
#tran22s = []
#
## The empty transition matrix for each year, ready for implementation
#tran22 = []
#
#tran22 = deepcopy(tran0)
#for i in range(len(pr22s)):
#    tran22 = implement(tran22, pr22s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran22s.append(tran22)
#    tran22 = deepcopy(tran0)
## print (tran22s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs22 = []
## empty reproductive vector for implementation of each year
#rpv22 = rs
#
#for i in range(len(tran22s)):
#    rpv22 = mul(rpv22, tran22s[i])
##   rpvs22.append(rpv22)
##   print (rpv22)
#frpv.append(rpv22)
#
#""" Fertility rate data in the 2000s (women start to bear children at 23) """
#
##GFR1_23 = [126.66, 124.84, 121.19, 93.17, 74.83, 38.91, 38.35, 
##           29, 23.69, 17.14, 11.8, 6.57, 6.58, 7.43, 4.18]
##GFR2_23 = [0.00, 17.79, 20.54, 25.3, 30.03, 34.2, 31.66, 34.49,
##           33.09, 32.61, 20.03, 21.85, 12.85, 11.52, 10.13]
##GFR3_23 = [0.00, 0.00, 3.45, 3.9, 3.09, 5.21, 3.84, 5, 3.21, 3.57,
##           4.4, 2.81, 3.51, 4.05, 2.66]
#        
## List of the transitional probability vector for parous states each year in 
## 2000-2014, for women start to bear children at 23 in 2000
#pr23s = []
## Each year's transitional probability vector for parous states, ready for implementation
#pr23 = []
#
##for i in range (15):
##    if (i == 0):
##        pr23.append(GFR1_23[i]/1000)
##        pr23.append(GFR2_23[i]/1000)
##        pr23.append(GFR3_23[i]/1000)
##    else:
##        product1, product2, product3 = 1, 1, 1
##        for j in range (i):
##            product1 *= (1 - GFR1_23[j]/1000)
##            product2 *= (1 - GFR2_23[j]/1000)
##            product3 *= (1 - GFR3_23[j]/1000)
##        pr23.append((GFR1_23[i]/1000)/product1)
##        pr23.append((GFR2_23[i]/1000)/product2)
##        pr23.append((GFR3_23[i]/1000)/product3)
##    pr23s.append(pr23)
##    pr23 = []
#pr23s = deepcopy(pr22s[1:])
#pr23s[0][1] = 0.0
#pr23s[1][2] = 0.0
##for i in pr23s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 23
#tran23s = []
#
## The empty transition matrix for each year, ready for implementation
#tran23 = []
#
#tran23 = deepcopy(tran0)
#for i in range(len(pr23s)):
#    tran23 = implement(tran23, pr23s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran23s.append(tran23)
#    tran23 = deepcopy(tran0)
## print (tran23s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs23 = []
## empty reproductive vector for implementation of each year
#rpv23 = rs
#
#for i in range(len(tran23s)):
#    rpv23 = mul(rpv23, tran23s[i])
##   rpvs23.append(rpv23)
##   print (rpv23)
#frpv.append(rpv23)
#
#""" Fertility rate data in the 2000s (women start to bear children at 24) """
#
##GFR1_24 = [124.84, 103.17, 93.17, 72.28, 53.91, 26.51, 25.89,
##           27.78, 22.76, 15.41, 9.66, 5.37, 6.52, 5.23, 2.83]
##GFR2_24 = [0.00, 21.83, 25.3, 27.66, 32.78, 35.35, 35.59, 32.45,
##           28.37, 23.14, 17.81, 10.32, 14.92, 9.7, 5.77]
##GFR3_24 = [0.00, 0.00, 3.9, 4.35, 3.29, 5.23, 3.29, 6.08, 3.5, 
##           2.9, 4.65, 1.09, 2.08, 1.84, 2.95]
#
## List of the transitional probability vector for parous states each year in 
## 2000-2014, for women start to bear children at 24 in 2000
#pr24s = []
## Each year's transitional probability vector for parous states, ready for implementation
#pr24 = []
#
##for i in range (15):
##    if (i == 0):
##        pr24.append(GFR1_24[i]/1000)
##        pr24.append(GFR2_24[i]/1000)
##        pr24.append(GFR3_24[i]/1000)
##    else:
##        product1, product2, product3 = 1, 1, 1
##        for j in range (i):
##            product1 *= (1 - GFR1_24[j]/1000)
##            product2 *= (1 - GFR2_24[j]/1000)
##            product3 *= (1 - GFR3_24[j]/1000)
##        pr24.append((GFR1_24[i]/1000)/product1)
##        pr24.append((GFR2_24[i]/1000)/product2)
##        pr24.append((GFR3_24[i]/1000)/product3)
##    pr24s.append(pr24)
##    pr24 = []
#pr24s = deepcopy(pr23s[1:])
#pr24s[0][1] = 0.0
#pr24s[1][2] = 0.0
##for i in pr24s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 24
#tran24s = []
#
## The empty transition matrix for each year, ready for implementation
#tran24 = []
#
#tran24 = deepcopy(tran0)
#for i in range(len(pr24s)):
#    tran24 = implement(tran24, pr24s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran24s.append(tran24)
#    tran24 = deepcopy(tran0)
## print (tran24s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs24 = []
## empty reproductive vector for implementation of each year
#rpv24 = rs
#
#for i in range(len(tran24s)):
#    rpv24 = mul(rpv24, tran24s[i])
##   rpvs24.append(rpv24)
##   print (rpv24)
#frpv.append(rpv24)
#
#""" Fertility rate data in the 2000s (women start to bear children at 25) """
#
##GFR1_25 = [103.17, 76.47, 72.28, 49.18, 37.16, 17.86, 21.24, 19.84,
##           18.06, 12.18, 7.82, 3.25, 4.66, 4.13, 1.6]
##GFR2_25 = [0.00, 24.64, 27.66, 31.43, 34.74, 34.5, 35.36, 30.85, 26.05,
##           22.89, 14.64, 9.22, 8.97, 7.42, 5.72]
##GFR3_25 = [0.00, 0.00, 4.35, 3.81, 3.41, 4.86, 3.42, 4.06, 3.76, 2.44, 
##           4, 2.32, 3.2, 2.57, 1.72]
#        
## List of the transitional probability vector for parous states each year in 
## 2000-2014, for women start to bear children at 25 in 2000
#pr25s = []
## Each year's transitional probability vector for parous states, ready for implementation
#pr25 = []
#
##for i in range (15):
##    if (i == 0):
##        pr25.append(GFR1_25[i]/1000)
##        pr25.append(GFR2_25[i]/1000)
##        pr25.append(GFR3_25[i]/1000)
##    else:
##        product1, product2, product3 = 1, 1, 1
##        for j in range (i):
##            product1 *= (1 - GFR1_25[j]/1000)
##            product2 *= (1 - GFR2_25[j]/1000)
##            product3 *= (1 - GFR3_25[j]/1000)
##        pr25.append((GFR1_25[i]/1000)/product1)
##        pr25.append((GFR2_25[i]/1000)/product2)
##        pr25.append((GFR3_25[i]/1000)/product3)
##    pr25s.append(pr25)
##    pr25 = []
#pr25s = deepcopy(pr24s[1:])
#pr25s[0][1] = 0.0
#pr25s[1][2] = 0.0
##for i in pr25s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 25
#tran25s = []
#
## The empty transition matrix for each year, ready for implementation
#tran25 = []
#
#tran25 = deepcopy(tran0)
#for i in range(len(pr25s)):
#    tran25 = implement(tran25, pr25s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran25s.append(tran25)
#    tran25 = deepcopy(tran0)
## print (tran25s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs25 = []
## empty reproductive vector for implementation of each year
#rpv25 = rs
#
#for i in range(len(tran25s)):
#    rpv25 = mul(rpv25, tran25s[i])
##   rpvs25.append(rpv25)
##   print (rpv25)
#frpv.append(rpv25)
#
#""" Fertility rate data in the 2000s (women start to bear children at 26) """
#
##GFR1_26 = [76.47, 52.76, 49.18, 33.53, 24.91, 12.31, 13.18, 14.57, 13.84,
##           9.45, 6.42, 2.93, 3.53, 3.68, 0.94]
##GFR2_26 = [0.00, 26.37, 31.43, 31.56, 34.06, 33.81, 32.23, 25.14, 21.08,
##           18.18, 12.62, 8.54, 8.82, 6.85, 3.99]
##GFR3_26 = [0.00, 0.00, 3.81, 3.96, 3.24, 5.02, 2.75, 2.96, 3.00, 1.79,
##           3.64, 2.59, 2.33, 1.86, 1.05]
#        
## List of the transitional probability vector for parous states each year in 
## 2000-2014, for women start to bear children at 26 in 2000
#pr26s = []
## Each year's transitional probability vector for parous states, ready for implementation
#pr26 = []
#
##for i in range (15):
##    if (i == 0):
##        pr26.append(GFR1_26[i]/1000)
##        pr26.append(GFR2_26[i]/1000)
##        pr26.append(GFR3_26[i]/1000)
##    else:
##        product1, product2, product3 = 1, 1, 1
##        for j in range (i):
##            product1 *= (1 - GFR1_26[j]/1000)
##            product2 *= (1 - GFR2_26[j]/1000)
##            product3 *= (1 - GFR3_26[j]/1000)
##        pr26.append((GFR1_26[i]/1000)/product1)
##        pr26.append((GFR2_26[i]/1000)/product2)
##        pr26.append((GFR3_26[i]/1000)/product3)
##    pr26s.append(pr26)
##    pr26 = []
#pr26s = deepcopy(pr25s[1:])
#pr26s[0][1] = 0.0
#pr26s[1][2] = 0.0
##for i in pr26s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 26
#tran26s = []
#
## The empty transition matrix for each year, ready for implementation
#tran26 = []
#
#tran26 = deepcopy(tran0)
#for i in range(len(pr26s)):
#    tran26 = implement(tran26, pr26s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran26s.append(tran26)
#    tran26 = deepcopy(tran0)
## print (tran26s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs26 = []
## empty reproductive vector for implementation of each year
#rpv26 = rs
#
#for i in range(len(tran26s)):
#    rpv26 = mul(rpv26, tran26s[i])
##   rpvs26.append(rpv26)
##   print (rpv26)
#frpv.append(rpv26)
#
#""" Fertility rate data in the 2000s (women start to bear children at 27) """
#
##GFR1_27 = [52.76, 34.11, 33.53, 23.98, 17.55, 7.71, 8.44, 10.77, 9.16,
##           6.61, 5.36, 2.76, 3.28, 2.61, 1.4]
##GFR2_27 = [0.00, 27.96, 31.56, 30, 32.78, 28.28, 25.29, 22.17, 18.52,
##           13.54, 10.09, 6.66, 5.27, 3.64, 2.21]
##GFR3_27 = [0.00, 0.00, 3.96, 4.6, 3.03, 4.32, 3.72, 3.2, 1.98, 1.86,
##           3.21, 2.12, 1.34, 1.57, 1.3]
#        
## List of the transitional probability vector for parous states each year in 
## 2000-2014, for women start to bear children at 27 in 2000
#pr27s = []
## Each year's transitional probability vector for parous states, ready for implementation
#pr27 = []
#
##for i in range (15):
##    if (i == 0):
##        pr27.append(GFR1_27[i]/1000)
##        pr27.append(GFR2_27[i]/1000)
##        pr27.append(GFR3_27[i]/1000)
##    else:
##        product1, product2, product3 = 1, 1, 1
##        for j in range (i):
##            product1 *= (1 - GFR1_27[j]/1000)
##            product2 *= (1 - GFR2_27[j]/1000)
##            product3 *= (1 - GFR3_27[j]/1000)
##        pr27.append((GFR1_27[i]/1000)/product1)
##        pr27.append((GFR2_27[i]/1000)/product2)
##        pr27.append((GFR3_27[i]/1000)/product3)
##    pr27s.append(pr27)
##    pr27 = []
#pr27s = deepcopy(pr26s[1:])
#pr27s[0][1] = 0.0
#pr27s[1][2] = 0.0
##for i in pr27s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 27
#tran27s = []
#
## The empty transition matrix for each year, ready for implementation
#tran27 = []
#
#tran27 = deepcopy(tran0)
#for i in range(len(pr27s)):
#    tran27 = implement(tran27, pr27s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran27s.append(tran27)
#    tran27 = deepcopy(tran0)
## print (tran27s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs27 = []
## empty reproductive vector for implementation of each year
#rpv27 = rs
#
#for i in range(len(tran27s)):
#    rpv27 = mul(rpv27, tran27s[i])
##   rpvs27.append(rpv27)
##   print (rpv27)
#frpv.append(rpv27)
#
#""" Fertility rate data in the 2000s (women start to bear children at 28) """
#
##GFR1_28 = [34.11, 21.13, 23.98, 15.96, 12.42, 5.46, 8.32, 8.62, 11.56, 
##           7.44, 4.57, 1.54, 3.33, 1.97, 0.81]
##GFR2_28 = [0.00, 27.5, 30, 28.08, 28.08, 22.9, 20.03, 18.7, 12.57, 11.51,
##           8.11, 4.26, 6.05, 2.54, 2.94]
##GFR3_28 = [0.00, 0.00, 4.6, 3.97, 2.5, 3.76, 1.56, 1.95, 2.18, 2.37, 2.76,
##           0.88, 1.73, 0.6, 1.12]
#        
## List of the transitional probability vector for parous states each year in 
## 2000-2014, for women start to bear children at 28 in 2000
#pr28s = []
## Each year's transitional probability vector for parous states, ready for implementation
#pr28 = []
#
##for i in range (15):
##    if (i == 0):
##        pr28.append(GFR1_28[i]/1000)
##        pr28.append(GFR2_28[i]/1000)
##        pr28.append(GFR3_28[i]/1000)
##    else:
##        product1, product2, product3 = 1, 1, 1
##        for j in range (i):
##            product1 *= (1 - GFR1_28[j]/1000)
##            product2 *= (1 - GFR2_28[j]/1000)
##            product3 *= (1 - GFR3_28[j]/1000)
##        pr28.append((GFR1_28[i]/1000)/product1)
##        pr28.append((GFR2_28[i]/1000)/product2)
##        pr28.append((GFR3_28[i]/1000)/product3)
##    pr28s.append(pr28)
##    pr28 = []
#pr28s = deepcopy(pr27s[1:])
#pr28s[0][1] = 0.0
#pr28s[1][2] = 0.0
##for i in pr28s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 28
#tran28s = []
#
## The empty transition matrix for each year, ready for implementation
#tran28 = []
#
#tran28 = deepcopy(tran0)
#for i in range(len(pr28s)):
#    tran28 = implement(tran28, pr28s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran28s.append(tran28)
#    tran28 = deepcopy(tran0)
## print (tran28s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs28 = []
## empty reproductive vector for implementation of each year
#rpv28 = rs
#
#for i in range(len(tran28s)):
#    rpv28 = mul(rpv28, tran28s[i])
##   rpvs28.append(rpv28)
##   print (rpv28)
#frpv.append(rpv28)
#
#""" Fertility rate data in the 2000s (women start to bear children at 29) """
#
##GFR1_29 = [21.13, 13.27, 15.96, 9.81, 8.5, 4.19, 7.5, 9.34, 7.72, 5.47, 3.67,
##           1.73, 1.14, 1.21, 0.57]
##GFR2_29 = [0.00, 26.4, 28.08, 24.13, 20.95, 17.38, 15.42, 13.71, 10.07, 7.58, 
##           6.03, 2.3, 3.43, 2.14, 1.14]
##GFR3_29 = [0.00, 0.00, 3.97, 3.31, 2.28, 3.3, 3.14, 1.92, 1.94, 2.31, 2.18, 
##           1.17, 1.23, 0.52, 0.76]
#        
## List of the transitional probability vector for parous states each year in 
## 2000-2014, for women start to bear children at 29 in 2000
#pr29s = []
## Each year's transitional probability vector for parous states, ready for implementation
#pr29 = []
#
##for i in range (15):
##    if (i == 0):
##        pr29.append(GFR1_29[i]/1000)
##        pr29.append(GFR2_29[i]/1000)
##        pr29.append(GFR3_29[i]/1000)
##    else:
##        product1, product2, product3 = 1, 1, 1
##        for j in range (i):
##            product1 *= (1 - GFR1_29[j]/1000)
##            product2 *= (1 - GFR2_29[j]/1000)
##            product3 *= (1 - GFR3_29[j]/1000)
##        pr29.append((GFR1_29[i]/1000)/product1)
##        pr29.append((GFR2_29[i]/1000)/product2)
##        pr29.append((GFR3_29[i]/1000)/product3)
##    pr29s.append(pr29)
##    pr29 = []
#pr29s = deepcopy(pr28s[1:])
#pr29s[0][1] = 0.0
#pr29s[1][2] = 0.0
##for i in pr29s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 29
#tran29s = []
#
## The empty transition matrix for each year, ready for implementation
#tran29 = []
#
#tran29 = deepcopy(tran0)
#for i in range(len(pr29s)):
#    tran29 = implement(tran29, pr29s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran29s.append(tran29)
#    tran29 = deepcopy(tran0)
## print (tran29s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs29 = []
## empty reproductive vector for implementation of each year
#rpv29 = rs
#
#for i in range(len(tran29s)):
#    rpv29 = mul(rpv29, tran29s[i])
##   rpvs29.append(rpv29)
##   print (rpv29)
#frpv.append(rpv29)
#
#""" Fertility rate data in the 2000s (women start to bear children at 30) """
#
##GFR1_30 = [13.27, 7.99, 9.81, 5.58, 6.43, 2.66, 5.01, 5.95, 8.75, 6.18, 3.55,
##           1.58, 1.7, 1.06, 0.09]
##GFR2_30 = [0.00, 24.07, 24.13, 19.16, 16.95, 12.92, 11.16, 12.58, 9.9, 5.5,
##           5.12, 2.81, 1.95, 2.19, 1.04]
##GFR3_30 = [0.00, 0.00, 3.31, 3.47, 2.01, 2.68, 2.53, 2.46, 1.26, 1.51, 2.14, 
##           0.59, 1.49, 1.01, 0.57]
#        
## List of the transitional probability vector for parous states each year in 
## 2000-2014, for women start to bear children at 30 in 2000
#pr30s = []
## Each year's transitional probability vector for parous states, ready for implementation
#pr30 = []
#
##for i in range (15):
##    if (i == 0):
##        pr30.append(GFR1_30[i]/1000)
##        pr30.append(GFR2_30[i]/1000)
##        pr30.append(GFR3_30[i]/1000)
##    else:
##        product1, product2, product3 = 1, 1, 1
##        for j in range (i):
##            product1 *= (1 - GFR1_30[j]/1000)
##            product2 *= (1 - GFR2_30[j]/1000)
##            product3 *= (1 - GFR3_30[j]/1000)
##        pr30.append((GFR1_30[i]/1000)/product1)
##        pr30.append((GFR2_30[i]/1000)/product2)
##        pr30.append((GFR3_30[i]/1000)/product3)
##    pr30s.append(pr30)
##    pr30 = []
#
#pr30s = deepcopy(pr29s[1:])
#pr30s[0][1] = 0.0
#pr30s[1][2] = 0.0
##for i in pr30s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 30
#tran30s = []
#
## The empty transition matrix for each year, ready for implementation
#tran30 = []
#
#tran30 = deepcopy(tran0)
#for i in range(len(pr30s)):
#    tran30 = implement(tran30, pr30s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran30s.append(tran30)
#    tran30 = deepcopy(tran0)
## print (tran30s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs30 = []
## empty reproductive vector for implementation of each year
#rpv30 = rs
#
#for i in range(len(tran30s)):
#    rpv30 = mul(rpv30, tran30s[i])
##   rpvs30.append(rpv30)
##   print (rpv30)
#frpv.append(rpv30)
#
#""" Fertility rate data in the 2000s (women start to bear children at 31) """
#
#pr31s = deepcopy(pr30s[1:])
#pr31s[0][1] = 0.0
#pr31s[1][2] = 0.0
##for i in pr31s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 31
#tran31s = []
#
## The empty transition matrix for each year, ready for implementation
#tran31 = []
#
#tran31 = deepcopy(tran0)
#for i in range(len(pr31s)):
#    tran31 = implement(tran31, pr31s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran31s.append(tran31)
#    tran31 = deepcopy(tran0)
## print (tran31s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs31 = []
## empty reproductive vector for implementation of each year
#rpv31 = rs
#
#for i in range(len(tran31s)):
#    rpv31 = mul(rpv31, tran31s[i])
##   rpvs30.append(rpv31)
##   print (rpv31)
#frpv.append(rpv31)
#
#""" Fertility rate data in the 2000s (women start to bear children at 32) """
#
#pr32s = deepcopy(pr31s[1:])
#pr32s[0][1] = 0.0
#pr32s[1][2] = 0.0
##for i in pr32s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 32
#tran32s = []
#
## The empty transition matrix for each year, ready for implementation
#tran32 = []
#
#tran32 = deepcopy(tran0)
#for i in range(len(pr32s)):
#    tran32 = implement(tran32, pr32s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran32s.append(tran32)
#    tran32 = deepcopy(tran0)
## print (tran32s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs32 = []
## empty reproductive vector for implementation of each year
#rpv32 = rs
#
#for i in range(len(tran32s)):
#    rpv32 = mul(rpv32, tran32s[i])
##   rpvs30.append(rpv32)
##   print (rpv32)
#frpv.append(rpv32)
#
#""" Fertility rate data in the 2000s (women start to bear children at 33) """
#
#pr33s = deepcopy(pr32s[1:])
#pr33s[0][1] = 0.0
#pr33s[1][2] = 0.0
##for i in pr33s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 33
#tran33s = []
#
## The empty transition matrix for each year, ready for implementation
#tran33 = []
#
#tran33 = deepcopy(tran0)
#for i in range(len(pr33s)):
#    tran33 = implement(tran33, pr33s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran33s.append(tran33)
#    tran33 = deepcopy(tran0)
## print (tran33s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs33 = []
## empty reproductive vector for implementation of each year
#rpv33 = rs
#
#for i in range(len(tran33s)):
#    rpv33 = mul(rpv33, tran33s[i])
##   rpvs30.append(rpv33)
##   print (rpv33)
#frpv.append(rpv33)
#
#""" Fertility rate data in the 2000s (women start to bear children at 34) """
#
#pr34s = deepcopy(pr33s[1:])
#pr34s[0][1] = 0.0
#pr34s[1][2] = 0.0
##for i in pr33s:
##    print (i, '\n')
#
## List of transitional matrices in 2000s for women start to bear children at 34
#tran34s = []
#
## The empty transition matrix for each year, ready for implementation
#tran34 = []
#
#tran34 = deepcopy(tran0)
#for i in range(len(pr34s)):
#    tran34 = implement(tran34, pr34s[i], pb00s_1, pb00s_2, pb00s_3)
#    tran34s.append(tran34)
#    tran34 = deepcopy(tran0)
## print (tran34s)
#    
## List of reproductive probabilities vectors for each year  
#rpvs34 = []
## empty reproductive vector for implementation of each year
#rpv34 = rs
#
#for i in range(len(tran34s)):
#    rpv34 = mul(rpv34, tran34s[i])
##   rpvs30.append(rpv34)
##   print (rpv34)
#frpv.append(rpv34)
#
##for i in frpv:
##    for j in range(len(i)):
##        i[j] = round(i[j], 5)
#
#for i in frpv:
#    print (i, '\n')
#
#fig00_rps = plt.gcf()
#fig00_rps.set_size_inches(18.5,10.5)
#x = [x for x in range (1,14)];
#plt.ylim(-0.1,1,0.1)
#plt.yticks(np.arange(0, 1.1, 0.1))
#cgraph00 = []
#for i in range (len(frpv)):
#    g, = plt.plot(x, frpv[i], 'bs', color=color[i], markersize = 10)
##    for j in range(len(frpv[i])):                                         
##        plt.annotate(str(round(frpv[i][j],2)), xy=(x[j]+0.1,frpv[i][j]+0.01*x[j]),
##                     horizontalalignment='left',verticalalignment='bottom')
#    cgraph00.append(g)
#fig00_rps.suptitle('Reproductive Probability Vectors of Women Starting Childbearing from 20-34', fontsize = 28)
#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
#plt.xlabel('Reproductive States', fontsize = 24)
#plt.ylabel('Probabilities in Each State', fontsize = 24)
#plt.legend([i for i in cgraph00], ['20', '21', '22', '23', '24', '25',
#      '26', '27','28', '29', '30', '31', '32', '33', '34'], 
#       loc='center left', bbox_to_anchor=(1, 0.5))
#plt.grid()
#plt.show()
##fig00_rps.savefig('rps00s.pdf', dpi=100)
#fig00_rps.savefig('rps00s.png', dpi=100)
#
## -----------------------------------------------------------------------------
## *****************************************************************************
## End of reproductive behavior simulation
#
#"""Second part:
#   Breast cancer incident rate change based on the change of reproductive
#   behaviors' changes among Chinese females
#"""
## -----------------------------------------------------------------------------
#"""For this part, we apply the idea of "relative risk index" to calculate the 
#   increasing incidence rate of breast cancer among the population. Each 
#   reproductive will have a corresponding relative risk index to represent the 
#   chance of women in that states developing a breast cancer in her lifetime.
#   Relative risk of breast cancer is reduced by 4.3% (95% CI 2.9-5.8) for each
#   year that a woman breastfeeds, in addition to a reduction of 7.0% (5.9-9.0)
#   for each birth.
#"""
### Implementation of index vector for the thirteen different reproductive stages
### The trend in the relative risk of breast cancer with increasing duration of
### breastfeeding is calculated. In such instances, the duration of breastfeeding
### associated with a particular category is taken to be the median duration within
### that category.
### Assumption: women with no child who had never breastfed are taken to have a 
### relative risk of 1.0.
#risk_index = [1.0,0,0,0,0,0,0,0,0,0,0,0,0]
#for i in range (13):
#    if (0 < i < 4):
#        risk_index[i] = risk_index[i-1]*(1-0.07)
#    elif (3 < i < 7):
#        risk_index[i] = risk_index[i-3]*(1-0.0215)
#    elif (6 < i < 10):
#        risk_index[i] = risk_index[i-3]*(1-0.043)
#    elif (9 < i < 13):
#        risk_index[i] = risk_index[i-3]*(1-0.043)
##print (risk_index) 
#        
#""" For each year in 2000s, based on the women's reproductive state, we are
#    gonna use the relative risk index to calculate their relative risk of 
#    developing breast cancer
#"""
#
#sum_ = []
#for i in frpv:
#    rri_sum = 0
#    for j in range (len(i)):
#        rri_sum += i[j]*risk_index[j]
#    sum_.append(rri_sum)
#print (sum_)
#
#fig00_rri = plt.gcf()
#fig00_rri.set_size_inches(18.5,10.5)
#x = [x for x in range (20,35)];
#plt.ylim(0.88,1)
#g, = plt.plot(x, sum_, 'g^', color=color[1], markersize = 15)
#for j in range(len(sum_)):                                         
#    plt.annotate(str(round(sum_[j],3)), xy=(x[j]+0.001,sum_[j]+0.001),
#                     horizontalalignment='left',verticalalignment='bottom')
#fig00_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 20-34', fontsize = 28)
#plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
#plt.yticks(np.arange(0.88, 1.01, 0.01))
#plt.grid()
#plt.xlabel('Ages for Starting the Childbearing Process', fontsize = 24)
#plt.ylabel('Relative Risk Index', fontsize = 24)
##plt.legend([i for i in cgraph00], ['20', '21', '22', '23', '24', '25',
##      '26', '27','28', '29', '30'], loc='center left', bbox_to_anchor=(1, 0.5))
#plt.show()
##fig00_rps.savefig('rps00s.pdf', dpi=100)
#fig00_rri.savefig('rri.png', dpi=100)
# -----------------------------------------------------------------------------
























