""" EEB C219B Project Code Version 2.0
    The Effects of Reproductive Behaviors on the Incidence of
    Female Breast Cancer in China
"""

"""In this version, force everyone to bear children at a certain age."""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from numpy import matrix

# Colors for graphing
color = ['black', 'blue', 'brown', 'gold', 'cyan', 'darkgray',
         'gray', 'green', 'lavender','darkred','darkgoldenrod',
         'burlywood', 'deeppink', 'chocolate', 'indigo','darkorange',
         'lime', 'darkkhaki', 'silver', 'darksage', 'goldenrod', 'thistle',
         'orchid', 'crimson', 'yellow', 'navy', 'seagreen', 'k', 'teal',
         'plum', 'olive', 'c', 'cornsilk']

# Problem setup
#------------------------------------------------------------------------------

# The probability vector for the 13 different reproductive states
# Every decade the vector will be reset to the initial value as below:
rs = matrix([[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
# Women in childbearing age who haven't given birth yet

""" Corresponing reproductive states are
    stable[1,0],stable[2,0],stable[3,0] 
[0, 0],[1,  0],  [2,  0],  [3,  0]
    stable[1,12],stable[2,12],stable[3,12]
       [1, <12], [2, <12], [3, <12]
stable[1,12-24],stable[2,12-24],stable[3,12-24]
       [1,12~24],[2,12~24],[3,12~24]
       [1, >24], [2, >24], [3, >24]
where the first entry represents the number of full term pregnancy,
and the second entry represents the breastfeeding duration.
"""

# Transition matrix setup
#------------------------------------------------------------------------------
from copy import deepcopy

# Initial transition matrix with all the entries to be zero
tran00 = matrix(np.zeros((22,22)))

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
    for i in range(0, tran.shape[0]):
        # In our setup, reproductive stages are sequential. 
        # So women can't go back to the stages they've been through, and can only
        # go to the states next to the states they stay in.
        # Here, implement the transition probabilities to other states first,
        # and the probability of staying at the same state will be the (1-sum(transitional probabilities))
        for j in range(0, tran.shape[1]):
            # from [0,0] state, it can only go to [1,0]
            if (i == 0):
                if (j == 1):
                    tran[i,j] = pr[0] * pb1[3]
                elif (j == 4):
                    tran[i,j] = pr[0] * (1 - pb1[3])
            # from the states in the most right column, they can only goes down one by one
            # and the transitional probabilities will be the same as the breastfeeding duration time change probabilities
            if (i == 3 or i == 6 or i == 9):
                if j == i + 3:
                    tran[i,j] = pb3[(i//3)-1]
                elif j == i + 12:
                    tran[i,j] = 1 - pb3[(i//3)-1]
            # from all the intermediate states, there are three paths for change
            if (i == 1):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    tran[i,j] = (1 - (1 - pr[1]) * pb1[3]) * pr[1]*pb2[3]
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    tran[i,j] = (1 - (1 - pr[1]) * (pb1[3])) * pb1[0]*(1-pr[1])
                elif j == i + 4:
                    tran[i,j] = (1 - (1 - pr[1]) * (pb1[3])) * pr[1]*pb2[0]
                elif j == i + 12:
                    tran[i,j] = (1 - pr[1]) * (pb1[3])
            if (i == 2):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    tran[i,j] = (1 - (1 - pr[2]) * (pb2[3])) * pr[2]*pb3[3]
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    tran[i,j] = (1 - (1 - pr[2]) * (pb2[3])) * pb2[0]*(1-pr[2])
                elif j == i + 4:
                    tran[i,j] = (1 - (1 - pr[2]) * (pb2[3])) * pr[2]*pb3[0]
                elif j == i + 12:
                    tran[i,j] = (1 - pr[2]) * (pb2[3])
            if (i == 4):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    tran[i,j] = (1 - (1 - pr[1]) * (1 - pb1[1])) * pr[1]*(1-pb2[1])
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    tran[i,j] = (1 - (1 - pr[1]) * (1 - pb1[1])) * pb1[1]*(1-pr[1])
                elif j == i + 4:
                    tran[i,j] = (1 - (1 - pr[1]) * (1 - pb1[1])) * pr[1]*pb2[1]
                elif j == i + 12:
                    tran[i,j] = (1 - pr[1]) * (1 - pb1[1])
            if (i == 5):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    tran[i,j] = (1 - (1 - pr[2]) * (1 - pb2[1])) * pr[2]*(1-pb3[1])
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    tran[i,j] = (1 - (1 - pr[2]) * (1 - pb2[1])) * pb2[1]*(1-pr[2])
                elif j == i + 4:
                    tran[i,j] = (1 - (1 - pr[2]) * (1 - pb2[1])) * pr[2]*pb3[1]
                elif j == i + 12:
                    tran[i,j] = (1 - pr[2]) * (1 - pb2[1])
            if (i == 7):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    tran[i,j] = (1 - (1 - pr[1]) * (1 - pb1[2])) * pr[1]*(1-pb2[2])
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    tran[i,j] = (1 - (1 - pr[1]) * (1 - pb1[2])) * pb1[2]*(1-pr[1])
                elif j == i + 4:
                    tran[i,j] = (1 - (1 - pr[1]) * (1 - pb1[2])) * pr[1]*pb2[2]
                elif j == i + 12:
                    tran[i,j] = (1 - pr[1]) * (1 - pb1[2])
            if (i == 8):
                # only parity changes, breastfeeding duration time stays the same
                if j == i + 1:
                    tran[i,j] = (1 - (1 - pr[2]) * (1 - pb2[2])) * pr[2]*(1-pb3[2])
                # only breastfeeding duration changes, parity stays the same
                elif j == i + 3:
                    tran[i,j] = (1 - (1 - pr[2]) * (1 - pb2[2])) * pb2[2]*(1-pr[2])
                elif j == i + 4:
                    tran[i,j] = (1 - (1 - pr[2]) * (1 - pb2[2])) * pr[2]*pb3[2]
                elif j == i + 12:
                    tran[i,j] = (1 - pr[2]) * (1 - pb2[2])
            if (i == 10 or i == 11):
                if j == i + 1:
                    tran[i,j] = pr[i%3]
            if (i == 13 or i == 14):
                if (j == i - 11 or j == i - 8):
                    tran[i,j] = pr[i%3]
                    if (j == i - 11 and i == 13):
                        tran[i,j] *= pb2[3]
                    elif (j == i - 8 and i == 13):
                        tran[i,j] *= pb2[0]
                    elif (j == i - 11 and i == 14):
                        tran[i,j] *= pb3[3]
                    elif (j == i - 8 and i == 14):
                        tran[i,j] *= pb3[0]
            if (i == 16 or i == 17):
                if (j == i - 8 or j == i - 11):
                    tran[i,j] = pr[i%3]
                    if (j == i - 11 and i == 16):
                        tran[i,j] *= (1 - pb2[1])
                    elif (j == i - 8 and i == 16):
                        tran[i,j] *= pb2[1]
                    elif (j == i - 11 and i == 17):
                        tran[i,j] *= (1 - pb3[1])
                    elif (j == i - 8 and i == 17):
                        tran[i,j] *= pb3[1]
            if (i == 19 or i == 20):
                if (j == i - 8 or j == i - 11):
                    tran[i,j] = pr[i%3]
                    if (j == i - 11 and i == 19):
                        tran[i,j] *= (1 - pb2[2])
                    elif (j == i - 8 and i == 19):
                        tran[i,j] *= pb2[2]
                    elif (j == i - 11 and i == 20):
                        tran[i,j] *= (1 - pb3[2])
                    elif (j == i - 8 and i == 20):
                        tran[i,j] *= pb3[2]
                
    # Implement the diagonal entries
    for k in range (tran.shape[0]):
        row_sum = 1 - tran[k,:].sum()
        tran[k,k] = row_sum

    return tran

"""Function for getting the reproductive probability vector in different conditions"""
def get_rpv(pr_, pb1, pb2, pb3):
    trans = []
    tran = np.copy(tran00)
    for i in range(len(pr_)):
        tran.fill(0.0)
        tran = np.copy(implement(tran, pr_[i], pb1, pb2, pb3))
        trans.append(tran)
        
    # List of reproductive probabilities vectors for each year  
    rpvs = []
    # empty reproductive vector for implementation of each year
    rpv = rs
    
    for i in range(len(trans)):
        rpv = rpv*trans[i]
    rpvs.append(rpv)
        
    for i in range(1,10):
        rpv[0,i] += rpv[0,i+12]

    final_rpv = deepcopy(rpv.getA1()[0:13])
    return final_rpv
    
"""Function for simulation of final reproductive vectors in each cohort. """
def completeSimulation(initialPr, pb1, pb2, pb3, trials):
    pr = initialPr
    pr[0][1] = 0.0
    pr[1][2] = 0.0
    frpv = []
    for i in range (trials):
        rpv = get_rpv(pr, pb1, pb2, pb3)
        frpv.append(rpv)
        if (len(pr) == 1):
            break
        pr = deepcopy(pr[1:])
        pr[0][1] = 0.0
        pr[1][2] = 0.0
        pr[0][2] = 0.0
#    print (pr)
    return frpv
  
""" Probability vector of having a child at exactly age of X.
    Every entry represents the percentage of female population bearing child at
    at a certain age. The last entry represents no children in whole life. """
def haveChildAt(GFR):
    PCN = []
    # assume that at 20, all female individual won't have any child
    for i in range(len(GFR)):
        product = 1
        for j in range (i):
            product *= (1 - GFR[j]/1000)
        PN = product * GFR[i]/1000
        PCN.append(PN)
    sum_ = sum(PCN)
    for i in range (len(PCN)):
        PCN[i] = PCN[i]/sum_
#    print (PCN)
    return PCN
    
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
P_noBreastfeeding = 0.2643

mu1, sigma1 = 13.62, 9.80
pb01 = norm(loc=mu1,scale=sigma1)
""" For every child a woman gives birth to, the probability of her devoting no 
    breastfeeding is pb01_0, probability of her devoting 0~12 months breastfeeding
    is pb01_12, and 12~24, more than 24 months breastfeeding are pb01_24, pb01_36. 
"""
pb01_0 = P_noBreastfeeding
pb01_12 = pb01.cdf(12)
pb01_24 = pb01.cdf(24)-pb01.cdf(12)
pb01_36 = 1-pb01.cdf(24)
pb00s_1 = [1-pb01_0, pb01_24, pb01_36, pb01_0]
#print (pb00s_1)

pb02_0 = pb01_0
pb00s_2 = [pb02_0, pb00s_1[0]**2, pb00s_1[1], 1-pb02_0]
#print (pb00s_2)

pb03_0 = pb01_0
pb00s_3 = [pb03_0, pb00s_1[0]**2, pb00s_1[0]**3, 1-pb03_0]
#print (pb00s_3)

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
        risk_index[i] = risk_index[i-3]*(1-0.043)
    elif (6 < i < 10):
        risk_index[i] = risk_index[i-3]*(1-0.043)
    elif (9 < i < 13):
        risk_index[i] = risk_index[i-3]*(1-0.043)
##print (risk_index) 

""" A list for final reproductive vectors of women who start to bear children
    at different ages  """    

GFR1_20 =  [42.93, 75.63, 94.76,	105.84, 105.60, 92.57, 78.06, 62.44, 48.73, 
            37.25, 25.85, 19.50, 14.01, 10.35, 8.08, 6.09, 4.88, 3.64, 3.09,
            2.66, 2.23, 1.94, 1.582, 1.47, 1.32, 1.26, 1.19, 1.268, 1.219, 1.2]
GFR2_20 =  [2.79, 5.57, 8.86,	14.12,	17.62, 22.102, 26.29, 28.75, 31.85,	31.60,
            30.15, 29.328, 25.73, 20.938, 17.11, 13.13, 9.886, 7.92, 5.864,
            4.31, 3.17, 2.24, 1.46,	1.244, 0.95, 0.85, 0.79,	0.72, 0.62, 0.59]
GFR3_20 =  [0.0, 0.25, 0.81, 1.4,	2.22, 2.59, 3.37, 3.73, 4.138, 4.102, 4.494,	
            4.36, 3.94, 3.71, 3.18, 2.644,	2.306,	2.1,	1.88, 1.25,
            1.012, 0.80, 0.77,	0.55, 0.47, 0.41, 0.32, 0.29, 0.298, 0.31]

fig00_gfr = plt.gcf()
fig00_gfr.set_size_inches(8,6)
x = [x for x in range (20, 50)];
plt.ylim(0, 110, 10)
#plt.yticks(np.arange(0, 1.1, 0.1))
g, = plt.plot(x, GFR1_20, color=color[3])
g, = plt.plot(x, GFR2_20, color=color[2])
g, = plt.plot(x, GFR3_20, color=color[1])
fig00_gfr.suptitle('GFR in 2000-2014 at different ages (Average)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
#fig00_rps.savefig('rps00s.pdf', dpi=100)
fig00_gfr.savefig('gfr00_7.0_Average_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 20) """

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr20s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr20 = []

for i in range (len(GFR1_20)):
    pr20.append(GFR1_20[i]/1000)
    pr20.append(GFR2_20[i]/1000)
    pr20.append(GFR3_20[i]/1000)
    pr20s.append(pr20)
    pr20 = []
#print (pr20s)

frpv = deepcopy(completeSimulation(pr20s, pb00s_1, pb00s_2, pb00s_3, 25))

fig00_rps = plt.gcf()
fig00_rps.set_size_inches(11,6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph00 = []
for i in range (len(frpv)):
    g, = plt.plot(x, frpv[i], 'bs', color=color[i], markersize = 10)
#    for j in range(len(frpv[i])):                                         
#        plt.annotate(str(round(frpv[i][j],2)), xy=(x[j]+0.1,frpv[i][j]+0.01*x[j]),
#                     horizontalalignment='left',verticalalignment='bottom')
    cgraph00.append(g)
fig00_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 20-44 (Average)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph00], ['20', '21', '22', '23', '24', '25', 
      '26', '27','28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
      '38', '39', '40', '41', '42', '43', '44'], 
       loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
#fig00_rps.savefig('rps00s.pdf', dpi=100)
fig00_rps.savefig('rps00_7.0_Average_China20s.png', dpi=100)
#
""" For each year in 2000s, based on the women's reproductive state, we are
    gonna use the relative risk index to calculate their relative risk of 
    developing breast cancer
"""
sum_ = []
for i in frpv:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum_.append(rri_sum)
#print (sum_)

fig00_rri = plt.gcf()
fig00_rri.set_size_inches(14,6)
x = [x for x in range (20,45)];
g, = plt.plot(x, sum_, 'g^', color=color[1], markersize = 15)
for j in range(len(sum_)):                                         
    plt.annotate(str(round(sum_[j],3)), xy=(x[j]+0.001,sum_[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig00_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 20-44 (Average)', fontsize = 20)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.ylim(0.86, 0.90, 0.01)
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
#fig00_rps.savefig('rps00s.pdf', dpi=100)
fig00_rri.savefig('rri_7.0_Average_China20s.png', dpi=100)
#
""" Probability of not having a child till X age.
    Every entry represents the percentage of female population bearing child at
    at a certain age. The last entry represents no children in whole life."""
#PCN = haveChildAt(GFR1_20)
#
#fig00_PCN = plt.gcf()
#fig00_PCN.set_size_inches(18.5, 10.5)
#x = [x for x in range (20, 50)]
#g, = plt.plot(x, PCN, color = color[1])
#fig00_PCN.suptitle('The Fraction of women who start their childbearing process at each age', fontsize = 28)
#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
#plt.xlabel('Ages', fontsize = 24)
#plt.ylabel('Percentage of the female population', fontsize = 24)
#plt.grid()
#plt.show()
#fig00_PCN.savefig('PCN00_7.0_Average_China20s.png', dpi=100)

"""match the individual to the whole populaiton of the average cohort"""
#pop_frpv = np.zeros(13)
#sum_frpv = 0
#for i in range(15):
#    for j in range (len(frpv[0])):
#        pop_frpv[j] += PCN[i] * frpv[i][j]
#pop_frpv[0] = PCN[(len(PCN)-1)]
##print(pop_frpv)
#
#fig00_p_rps = plt.gcf()
#fig00_p_rps.set_size_inches(18.5,10.5)
#x = [x for x in range (1,14)];
#plt.ylim(-0.1,1,0.1)
#plt.yticks(np.arange(0, 1.1, 0.1))
#cgraph00 = []
#g, = plt.plot(x, pop_frpv, 'bs', color=color[2], markersize = 10)
#for j in range(len(pop_frpv)):                                        
#    plt.annotate(str(round(pop_frpv[j],3)), xy=(x[j]+0.1,pop_frpv[j]+0.01*x[j]),
#                 horizontalalignment='left',verticalalignment='bottom')
#fig00_p_rps.suptitle('Average Cohort', fontsize = 28)
#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
#plt.xlabel('Reproductive States', fontsize = 24)
#plt.ylabel('Probabilities in Each State', fontsize = 24)
#plt.grid()
#plt.show()
##fig00_rps.savefig('rps00s.pdf', dpi=100)
#fig00_p_rps.savefig('rps00_7.0_Average_Pop_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 20) """

GFR1_20_ = [52.56, 90.82, 128.89, 137.67, 134.96, 93.96, 85.87, 77.87, 59.28, 
            44.81, 27.65, 20.40, 19.69, 17.79, 9.34, 8.16, 5, 4.18, 2.83, 1.6,
           0.94, 1.4, 0.81, 0.57, 0.09, 0.59, 0.33, 0.22, 0.2, 0.11]
GFR2_20_ = [3.16, 6.04, 7.99, 12.82, 14.17, 22.68, 26.39, 31.63,
           33.54, 32.35, 26.72, 24.53, 24.14, 22.30, 20.29, 16.57,
           11.09, 10.13, 5.77, 5.72, 3.99, 2.21, 2.94, 1.14, 1.04, 0.73,
           0.43, 0.67, 0.51, 0.42]
GFR3_20_ = [0.00, 0.57, 1.51, 2.87, 2.15, 3.05, 3.53, 2.76, 3.73,
           4.6, 5.42, 4.58, 4.97, 4.99, 5.35, 5.02, 2.84, 2.66, 2.95,
           1.72, 1.05, 1.3, 1.12, 0.76, 0.57, 0.45, 0.65, 0.33, 0.2, 0.11]
           
fig20_gfr = plt.gcf()
fig20_gfr.set_size_inches(8,6)
x = [x for x in range (20, 50)];
plt.ylim(0, 140, 10)
#plt.yticks(np.arange(0, 1.1, 0.1))
g, = plt.plot(x, GFR1_20_, color=color[3])
g, = plt.plot(x, GFR2_20_, color=color[2])
g, = plt.plot(x, GFR3_20_, color=color[1])
fig20_gfr.suptitle('GFR in 2000-2014 at different ages (20-year-old cohort)', fontsize = 16)
#plt.title('GFR in 2000-2014 at different ages (20-year-old cohort)', fontsize = 16, loc = 'center')
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
       loc='best')
plt.grid()
plt.show()
fig20_gfr.savefig('gfr00_7.0_20_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr20_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr20_ = []
for i in range (len(GFR1_20_)):
    pr20_.append(GFR1_20_[i]/1000)
    pr20_.append(GFR2_20_[i]/1000)
    pr20_.append(GFR3_20_[i]/1000)
    pr20_s.append(pr20_)
    pr20_ = []
#print (pr20s)

frpv_20 = deepcopy(completeSimulation(pr20_s, pb00s_1, pb00s_2, pb00s_3, 25))

fig20_rps = plt.gcf()
fig20_rps.set_size_inches(11,6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph20 = []
for i in range (len(frpv_20)):
    g, = plt.plot(x, frpv_20[i], 'bs', color=color[i], markersize = 10)
#    for j in range(len(frpv[i])):                                         
#        plt.annotate(str(round(frpv[i][j],2)), xy=(x[j]+0.1,frpv[i][j]+0.01*x[j]),
#                     horizontalalignment='left',verticalalignment='bottom')
    cgraph20.append(g)
fig20_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 20-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph20], ['20', '21', '22', '23', '24', '25',
      '26', '27','28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
      '38', '39', '40', '41', '42', '43', '44'], 
       loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
#fig00_rps.savefig('rps00s.pdf', dpi=100)
fig20_rps.savefig('rps00_7.0_20_China20s.png', dpi=100)

sum20 = []
for i in frpv_20:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum20.append(rri_sum)
#print (sum_)

fig20_rri = plt.gcf()
fig20_rri.set_size_inches(14,6)
x = [x for x in range (20, 45)];
g, = plt.plot(x, sum20, 'g^', color=color[1], markersize = 15)
for j in range(len(sum20)):                                         
    plt.annotate(str(round(sum20[j],3)), xy=(x[j]+0.001,sum20[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig20_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 20-44', fontsize = 20)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig20_rri.savefig('rri_7.0_20_China20s.png', dpi=100)

"""match the individual to the whole populaiton of the 20-year-old cohort"""
""" Probability of not having a child till X age """
#PCN_20 = haveChildAt(GFR1_20_)
#
#fig20_PCN = plt.gcf()
#fig20_PCN.set_size_inches(18.5, 10.5)
#x = [x for x in range (20, 50]
#g, = plt.plot(x, PCN_20, color = color[1])
#fig20_PCN.suptitle('The Fraction of women who start their childbearing process at each age (20-year-old cohort)', fontsize = 24)
#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
#plt.xlabel('Ages', fontsize = 24)
#plt.ylabel('Percentage of the female population', fontsize = 24)
#plt.grid()
#plt.show()
#fig20_PCN.savefig('PCN20_7.0_20_China20s.png', dpi=100)
#
#pop_frpv20 = np.zeros(13)
#sum_frpv = 0
#for i in range(25):
#    for j in range (len(frpv_20[0])):
#        pop_frpv20[j] += PCN_20[i] * frpv_20[i][j]
#pop_frpv20[0] = PCN_20[(len(PCN_20)-1)]
##print(pop_frpv)
#
#fig20_p_rps = plt.gcf()
#fig20_p_rps.set_size_inches(18.5,10.5)
#x = [x for x in range (1,14)];
#plt.ylim(-0.1,1,0.1)
#plt.yticks(np.arange(0, 1.1, 0.1))
#cgraph20 = []
#g, = plt.plot(x, pop_frpv20, 'bs', color=color[2], markersize = 10)
#for j in range(len(pop_frpv20)):                                        
#    plt.annotate(str(round(pop_frpv20[j],3)), xy=(x[j]+0.1,pop_frpv20[j]+0.01*x[j]),
#                 horizontalalignment='left',verticalalignment='bottom')
#fig20_p_rps.suptitle('20-year-old Cohort', fontsize = 28)
#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
#plt.xlabel('Reproductive States', fontsize = 24)
#plt.ylabel('Probabilities in Each State', fontsize = 24)
#plt.grid()
#plt.show()
##fig00_rps.savefig('rps00s.pdf', dpi=100)
#fig20_p_rps.savefig('rps00_7.0_20_Pop_China20s.png', dpi=100)

#           
""" Fertility rate data in the 2000s (women start to bear children at 21) """

GFR1_21_ = [90.82, 114.22, 137.67, 137.86, 120.4, 72.88, 69.92, 58.47, 47.99,
            30.51, 21.47, 13.4, 14.21, 14.67, 8.16, 5.00, 4.18, 2.83, 1.6, 
            0.94, 1.4, 0.81, 0.57, 0.09, 0.59, 0.33, 0.22, 0.2, 0.11]
GFR2_21_ = [6.04, 9.91, 12.82, 16.04, 18.54, 28.21, 27.17, 32.76, 34.1, 31.49,
            27.23, 19.71, 21.52, 16.83, 16.57, 11.09, 10.13, 5.77, 5.72, 3.99, 
            2.21, 2.94, 1.14, 1.04, 0.73, 0.43, 0.67, 0.51, 0.42]
GFR3_21_ = [0.00, 1.17, 2.87, 3.49, 2.77, 4.1, 2.45, 3.7, 4.82, 2.47, 5.1, 3.31,
            3.21, 2.59, 5.02, 2.84, 2.66, 2.95, 1.72, 1.05, 1.3, 1.12, 0.76, 
            0.57, 0.45, 0.65, 0.33, 0.2, 0.11]
           
fig21_gfr = plt.gcf()
fig21_gfr.set_size_inches(8,6)
x = [x for x in range (21, 50)];
plt.xlim(21,50)
plt.ylim(0, 140, 10)
g, = plt.plot(x, GFR1_21_, color=color[3])
g, = plt.plot(x, GFR2_21_, color=color[2])
g, = plt.plot(x, GFR3_21_, color=color[1])
fig21_gfr.suptitle('GFR in 2000-2014 at different ages (21-year-old cohort)', fontsize = 16)
#print (min(x))
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
       loc='best')
plt.grid()
plt.show()
#fig00_rps.savefig('rps00s.pdf', dpi=100)
fig21_gfr.savefig('gfr00_7.0_21_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr21_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr21_ = []

for i in range (len(GFR1_21_)):
    pr21_.append(GFR1_21_[i]/1000)
    pr21_.append(GFR2_21_[i]/1000)
    pr21_.append(GFR3_21_[i]/1000)
    pr21_s.append(pr21_)
    pr21_ = []

frpv_21 = deepcopy(completeSimulation(pr21_s, pb00s_1, pb00s_2, pb00s_3, 24))

fig21_rps = plt.gcf()
fig21_rps.set_size_inches(11,6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph21 = []
for i in range (len(frpv_21)):
    g, = plt.plot(x, frpv_21[i], 'bs', color=color[i], markersize = 10)
##    for j in range(len(frpv[i])):                                         
##        plt.annotate(str(round(frpv[i][j],2)), xy=(x[j]+0.1,frpv[i][j]+0.01*x[j]),
##                     horizontalalignment='left',verticalalignment='bottom')
    cgraph21.append(g)
fig21_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 21-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph21], ['21', '22', '23', '24', '25',
      '26', '27','28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
      '38', '39', '40', '41', '42', '43', '44'], 
       loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
#fig00_rps.savefig('rps00s.pdf', dpi=100)
fig21_rps.savefig('rps00_7.0_21_China20s.png', dpi=100)

sum21 = []
for i in frpv_21:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum21.append(rri_sum)
#print (sum_)

fig21_rri = plt.gcf()
fig21_rri.set_size_inches(14,6)
x = [x for x in range (21,45)];
g, = plt.plot(x, sum21, 'g^', color=color[1], markersize = 15)
for j in range(len(sum21)):                                         
    plt.annotate(str(round(sum21[j],3)), xy=(x[j]+0.001,sum21[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig21_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 21-44', fontsize = 20)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig21_rri.savefig('rri_7.0_21_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 22) """
GFR1_22_ = [114.22, 126.66, 137.86, 121.19, 102.24, 55.31, 48.66,
           44.29, 34.14, 24.05, 17.04, 9.93, 11.35, 10.35, 5.00, 4.18, 2.83, 1.6, 
           0.94, 1.4, 0.81, 0.57, 0.09, 0.59, 0.33, 0.22, 0.2, 0.11]

GFR2_22_ = [9.91, 14.00, 16.04, 20.54, 25.23, 31.07, 33.05, 33.61,
           33.26, 29.22, 25.92, 21.83, 16.20, 15.64, 11.09, 10.13, 5.77, 5.72,
           3.99, 2.21, 2.94, 1.14, 1.04, 0.73, 0.43, 0.67, 0.51, 0.42]
           
GFR3_22_ = [0.00, 2.06, 3.49, 3.45, 2.89, 4.41, 4.45, 1.55, 4.93,
           4.09, 5.46, 3.34, 4.32, 3.36, 2.84, 2.66, 2.95, 1.72, 1.05, 1.3, 
           1.12, 0.76, 0.57, 0.45, 0.65, 0.33, 0.2, 0.11]
           
fig22_gfr = plt.gcf()
fig22_gfr.set_size_inches(8,6)
x = [x for x in range (22, 50)];
plt.xlim(22,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_22_, color=color[3])
g, = plt.plot(x, GFR2_22_, color=color[2])
g, = plt.plot(x, GFR3_22_, color=color[1])
fig22_gfr.suptitle('GFR in 2000-2014 at different ages (22-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig22_gfr.savefig('gfr00_7.0_22_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr22_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr22_ = []

for i in range (len(GFR1_22_)):
    pr22_.append(GFR1_22_[i]/1000)
    pr22_.append(GFR2_22_[i]/1000)
    pr22_.append(GFR3_22_[i]/1000)
    pr22_s.append(pr22_)
    pr22_ = []

frpv_22 = deepcopy(completeSimulation(pr22_s, pb00s_1, pb00s_2, pb00s_3, 23))

fig22_rps = plt.gcf()
fig22_rps.set_size_inches(11,6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph22 = []
for i in range (len(frpv_22)):
    g, = plt.plot(x, frpv_22[i], 'bs', color=color[i], markersize = 10) 
#    if i == len(frpv_22) - 1:
#        for j in range (len(frpv_22[i])):                                   
#            plt.annotate(str(round(frpv_22[i][j],5)), xy=(x[j]+0.1,frpv_22[i][j]+0.01*x[j]),
#                horizontalalignment='left',verticalalignment='bottom') 
    cgraph22.append(g)
fig22_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 22-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph22], ['22', '23', '24', '25',
      '26', '27','28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
      '38', '39', '40', '41', '42', '43', '44'], 
       loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig22_rps.savefig('rps00_7.0_22_China20s.png', dpi=100)

sum22 = []
for i in frpv_22:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum22.append(rri_sum)
#print (sum22)

fig22_rri = plt.gcf()
fig22_rri.set_size_inches(14,6)
x = [x for x in range (22,45)];
g, = plt.plot(x, sum22, 'g^', color=color[1], markersize = 15)
for j in range(len(sum22)):                                         
    plt.annotate(str(round(sum22[j],3)), xy=(x[j]+0.001,sum22[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig22_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 22-44', fontsize = 20)
plt.xlim(22,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig22_rri.savefig('rri_7.0_22_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 23) """
#
GFR1_23_ = [126.66, 124.84, 121.19, 93.17, 74.83, 38.91, 38.35, 
           29, 23.69, 17.14, 11.8, 6.57, 6.58, 7.43, 4.18, 2.83, 1.6, 
           0.94, 1.4, 0.81, 0.57, 0.09, 0.59, 0.33, 0.22, 0.2, 0.11]
GFR2_23_ = [14, 17.79, 20.54, 25.3, 30.03, 34.2, 31.66, 34.49,
           33.09, 32.61, 20.03, 21.85, 12.85, 11.52, 10.13, 5.77, 5.72,
           3.99, 2.21, 2.94, 1.14, 1.04, 0.73, 0.43, 0.67, 0.51, 0.42]
GFR3_23_ = [0.00, 3.06, 3.45, 3.9, 3.09, 5.21, 3.84, 5, 3.21, 3.57,
           4.4, 2.81, 3.51, 4.05, 2.66, 2.95, 1.72, 1.05, 1.3, 
           1.12, 0.76, 0.57, 0.45, 0.65, 0.33, 0.2, 0.11]
           
fig23_gfr = plt.gcf()
fig23_gfr.set_size_inches(8,6)
x = [x for x in range (23, 50)];
plt.xlim(23,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_23_, color=color[3])
g, = plt.plot(x, GFR2_23_, color=color[2])
g, = plt.plot(x, GFR3_23_, color=color[1])
fig23_gfr.suptitle('GFR in 2000-2014 at different ages (23-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig23_gfr.savefig('gfr00_7.0_23_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr23_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr23_ = []

for i in range (len(GFR1_23_)):
    pr23_.append(GFR1_23_[i]/1000)
    pr23_.append(GFR2_23_[i]/1000)
    pr23_.append(GFR3_23_[i]/1000)
    pr23_s.append(pr23_)
    pr23_ = []

frpv_23 = deepcopy(completeSimulation(pr23_s, pb00s_1, pb00s_2, pb00s_3, 22))

fig23_rps = plt.gcf()
fig23_rps.set_size_inches(11,6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph23 = []
for i in range (len(frpv_23)):
    g, = plt.plot(x, frpv_23[i], 'bs', color=color[i], markersize = 10) 
    cgraph23.append(g)
fig23_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 23-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph23], ['23', '24', '25',
      '26', '27','28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
      '38', '39', '40', '41', '42', '43', '44'], 
       loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig23_rps.savefig('rps00_7.0_23_China20s.png', dpi=100)

sum23 = []
for i in frpv_23:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum23.append(rri_sum)
print (sum23)

fig23_rri = plt.gcf()
fig23_rri.set_size_inches(14,6)
x = [x for x in range (23,45)];
g, = plt.plot(x, sum23, 'g^', color=color[1], markersize = 15)
for j in range(len(sum23)):                                         
    plt.annotate(str(round(sum23[j],3)), xy=(x[j]+0.001,sum23[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig23_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 23-44', fontsize = 20)
plt.xlim(23,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig23_rri.savefig('rri_7.0_23_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 24) """
GFR1_24_ = [124.84, 103.17, 93.17, 72.28, 53.91, 26.51, 25.89,
           27.78, 22.76, 15.41, 9.66, 5.37, 6.52, 5.23, 2.83, 1.6, 
           0.94, 1.4, 0.81, 0.57, 0.09, 0.59, 0.33, 0.22, 0.2, 0.11]
GFR2_24_ = [17.79, 21.83, 25.3, 27.66, 32.78, 35.35, 35.59, 32.45,
           28.37, 23.14, 17.81, 10.32, 14.92, 9.7, 5.77, 5.72,
           3.99, 2.21, 2.94, 1.14, 1.04, 0.73, 0.43, 0.67, 0.51, 0.42]
GFR3_24_ = [0.00, 4, 3.9, 4.35, 3.29, 5.23, 3.29, 6.08, 3.5, 
           2.9, 4.65, 1.09, 2.08, 1.84, 2.95, 1.72, 1.05, 1.3, 
           1.12, 0.76, 0.57, 0.45, 0.65, 0.33, 0.2, 0.11]
           
fig24_gfr = plt.gcf()
fig24_gfr.set_size_inches(8,6)
x = [x for x in range (24, 50)];
plt.xlim(24,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_24_, color=color[3])
g, = plt.plot(x, GFR2_24_, color=color[2])
g, = plt.plot(x, GFR3_24_, color=color[1])
fig24_gfr.suptitle('GFR in 2000-2014 at different ages (24-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig24_gfr.savefig('gfr00_7.0_24_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr24_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr24_ = []

for i in range (len(GFR1_24_)):
    pr24_.append(GFR1_24_[i]/1000)
    pr24_.append(GFR2_24_[i]/1000)
    pr24_.append(GFR3_24_[i]/1000)
    pr24_s.append(pr24_)
    pr24_ = []

frpv_24 = deepcopy(completeSimulation(pr24_s, pb00s_1, pb00s_2, pb00s_3, 21))

fig24_rps = plt.gcf()
fig24_rps.set_size_inches(11,6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph24 = []
for i in range (len(frpv_24)):
    g, = plt.plot(x, frpv_24[i], 'bs', color=color[i], markersize = 10) 
    cgraph24.append(g)
fig24_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 24-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph24], ['24', '25',
      '26', '27','28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
      '38', '39', '40', '41', '42', '43', '44'], 
       loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig24_rps.savefig('rps00_7.0_24_China20s.png', dpi=100)

sum24 = []
for i in frpv_24:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum24.append(rri_sum)
#print (sum24)

fig24_rri = plt.gcf()
fig24_rri.set_size_inches(14,6)
x = [x for x in range (24,45)];
g, = plt.plot(x, sum24, 'g^', color=color[1], markersize = 15)
for j in range(len(sum24)):                                         
    plt.annotate(str(round(sum24[j],3)), xy=(x[j]+0.001,sum24[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig24_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 24-44', fontsize = 20)
plt.xlim(24,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig24_rri.savefig('rri_7.0_24_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 25) """

GFR1_25_ = [103.17, 76.47, 72.28, 49.18, 37.16, 17.86, 21.24, 19.84,
           18.06, 12.18, 7.82, 3.25, 4.66, 4.13, 1.6, 0.94, 1.4, 0.81, 0.57,
            0.09, 0.59, 0.33, 0.22, 0.2, 0.11]
GFR2_25_ = [21.83, 24.64, 27.66, 31.43, 34.74, 34.5, 35.36, 30.85, 26.05,
           22.89, 14.64, 9.22, 8.97, 7.42, 5.72, 3.99, 2.21, 2.94, 1.14, 
           1.04, 0.73, 0.43, 0.67, 0.51, 0.42]
GFR3_25_ = [0.00, 4.76, 4.35, 3.81, 3.41, 4.86, 3.42, 4.06, 3.76, 2.44, 
           4, 2.32, 3.2, 2.57, 1.72, 1.05, 1.3, 
           1.12, 0.76, 0.57, 0.45, 0.65, 0.33, 0.2, 0.11]

fig25_gfr = plt.gcf()
fig25_gfr.set_size_inches(8,6)
x = [x for x in range (25, 50)];
plt.xlim(25,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_25_, color=color[3])
g, = plt.plot(x, GFR2_25_, color=color[2])
g, = plt.plot(x, GFR3_25_, color=color[1])
fig25_gfr.suptitle('GFR in 2000-2014 at different ages (25-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig25_gfr.savefig('gfr00_7.0_25_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr25_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr25_ = []

for i in range (len(GFR1_25_)):
    pr25_.append(GFR1_25_[i]/1000)
    pr25_.append(GFR2_25_[i]/1000)
    pr25_.append(GFR3_25_[i]/1000)
    pr25_s.append(pr25_)
    pr25_ = []

frpv_25 = deepcopy(completeSimulation(pr25_s, pb00s_1, pb00s_2, pb00s_3, 20))

fig25_rps = plt.gcf()
fig25_rps.set_size_inches(11,6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph25 = []
for i in range (len(frpv_25)):
    g, = plt.plot(x, frpv_25[i], 'bs', color=color[i], markersize = 10) 
#    if i == len(frpv_24) - 1:
#        for j in range (len(frpv_22[i])):                                   
#            plt.annotate(str(round(frpv_22[i][j],5)), xy=(x[j]+0.1,frpv_22[i][j]+0.01*x[j]),
#                horizontalalignment='left',verticalalignment='bottom') 
    cgraph25.append(g)
fig25_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 25-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph25], ['25',
      '26', '27','28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
      '38', '39', '40', '41', '42', '43', '44'], 
      loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig25_rps.savefig('rps00_7.0_25_China20s.png', dpi=100)

sum25 = []
for i in frpv_25:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum25.append(rri_sum)
#print (sum24)

fig25_rri = plt.gcf()
fig25_rri.set_size_inches(14,6)
x = [x for x in range (25,45)];
g, = plt.plot(x, sum25, 'g^', color=color[1], markersize = 15)
for j in range(len(sum25)):                                         
    plt.annotate(str(round(sum25[j],3)), xy=(x[j]+0.001,sum25[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig25_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 25-44', fontsize = 20)
plt.xlim(25,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig25_rri.savefig('rri_7.0_25_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 26) """

GFR1_26_ = [76.47, 52.76, 49.18, 33.53, 24.91, 12.31, 13.18, 14.57, 13.84,
           9.45, 6.42, 2.93, 3.53, 3.68, 0.94, 1.4, 0.81, 0.57,
            0.09, 0.59, 0.33, 0.22, 0.2, 0.11]
GFR2_26_ = [24.64, 26.37, 31.43, 31.56, 34.06, 33.81, 32.23, 25.14, 21.08,
           18.18, 12.62, 8.54, 8.82, 6.85, 3.99, 2.21, 2.94, 1.14, 
           1.04, 0.73, 0.43, 0.67, 0.51, 0.42]
GFR3_26_ = [0.00, 5.23, 3.81, 3.96, 3.24, 5.02, 2.75, 2.96, 3.00, 1.79,
           3.64, 2.59, 2.33, 1.86, 1.05, 1.3, 
           1.12, 0.76, 0.57, 0.45, 0.65, 0.33, 0.2, 0.11]

fig26_gfr = plt.gcf()
fig26_gfr.set_size_inches(8,6)
x = [x for x in range (26, 50)];
plt.xlim(26,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_26_, color=color[3])
g, = plt.plot(x, GFR2_26_, color=color[2])
g, = plt.plot(x, GFR3_26_, color=color[1])
fig26_gfr.suptitle('GFR in 2000-2014 at different ages (26-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig26_gfr.savefig('gfr00_7.0_26_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr26_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr26_ = []

for i in range (len(GFR1_26_)):
    pr26_.append(GFR1_26_[i]/1000)
    pr26_.append(GFR2_26_[i]/1000)
    pr26_.append(GFR3_26_[i]/1000)
    pr26_s.append(pr26_)
    pr26_ = []

frpv_26 = deepcopy(completeSimulation(pr26_s, pb00s_1, pb00s_2, pb00s_3, 19))

fig26_rps = plt.gcf()
fig26_rps.set_size_inches(11,6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph26 = []
for i in range (len(frpv_26)):
    g, = plt.plot(x, frpv_26[i], 'bs', color=color[i], markersize = 10) 
#    if i == len(frpv_24) - 1:
#        for j in range (len(frpv_22[i])):                                   
#            plt.annotate(str(round(frpv_22[i][j],5)), xy=(x[j]+0.1,frpv_22[i][j]+0.01*x[j]),
#                horizontalalignment='left',verticalalignment='bottom') 
    cgraph26.append(g)
fig26_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 26-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph26], ['26', '27','28', '29', '30', '31', '32', 
'33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44'], 
       loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig26_rps.savefig('rps00_7.0_26_China20s.png', dpi=100)

sum26 = []
for i in frpv_26:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum26.append(rri_sum)
#print (sum24)

fig26_rri = plt.gcf()
fig26_rri.set_size_inches(14,6)
x = [x for x in range (26,45)];
g, = plt.plot(x, sum26, 'g^', color=color[1], markersize = 15)
for j in range(len(sum26)):                                         
    plt.annotate(str(round(sum26[j],3)), xy=(x[j]+0.001,sum26[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig26_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 26-44', fontsize = 20)
plt.xlim(26,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig26_rri.savefig('rri_7.0_26_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 27) """

GFR1_27_ = [52.76, 34.11, 33.53, 23.98, 17.55, 7.71, 8.44, 10.77, 9.16,
           6.61, 5.36, 2.76, 3.28, 2.61, 1.4, 0.81, 0.57,
            0.09, 0.59, 0.33, 0.22, 0.2, 0.11]
GFR2_27_ = [26.37, 27.96, 31.56, 30, 32.78, 28.28, 25.29, 22.17, 18.52,
           13.54, 10.09, 6.66, 5.27, 3.64, 2.21, 2.94, 1.14, 
           1.04, 0.73, 0.43, 0.67, 0.51, 0.42]
GFR3_27_ = [0.00, 5.29, 3.96, 4.6, 3.03, 4.32, 3.72, 3.2, 1.98, 1.86,
           3.21, 2.12, 1.34, 1.57, 1.3, 1.12, 0.76, 0.57, 0.45, 
           0.65, 0.33, 0.2, 0.11]

fig27_gfr = plt.gcf()
fig27_gfr.set_size_inches(8,6)
x = [x for x in range (27, 50)];
plt.xlim(27,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_27_, color=color[3])
g, = plt.plot(x, GFR2_27_, color=color[2])
g, = plt.plot(x, GFR3_27_, color=color[1])
fig27_gfr.suptitle('GFR in 2000-2014 at different ages (27-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig27_gfr.savefig('gfr00_7.0_27_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr27_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr27_ = []

for i in range (len(GFR1_27_)):
    pr27_.append(GFR1_27_[i]/1000)
    pr27_.append(GFR2_27_[i]/1000)
    pr27_.append(GFR3_27_[i]/1000)
    pr27_s.append(pr27_)
    pr27_ = []

frpv_27 = deepcopy(completeSimulation(pr27_s, pb00s_1, pb00s_2, pb00s_3, 18))

fig27_rps = plt.gcf()
fig27_rps.set_size_inches(11,6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph27 = []
for i in range (len(frpv_27)):
    g, = plt.plot(x, frpv_27[i], 'bs', color=color[i], markersize = 10) 
#    if i == len(frpv_24) - 1:
#        for j in range (len(frpv_22[i])):                                   
#            plt.annotate(str(round(frpv_22[i][j],5)), xy=(x[j]+0.1,frpv_22[i][j]+0.01*x[j]),
#                horizontalalignment='left',verticalalignment='bottom') 
    cgraph27.append(g)
fig27_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 27-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph27], ['27','28', '29', '30', '31', '32', 
'33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44'], 
       loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig27_rps.savefig('rps00_7.0_27_China20s.png', dpi=100)

sum27 = []
for i in frpv_27:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum27.append(rri_sum)
#print (sum24)

fig27_rri = plt.gcf()
fig27_rri.set_size_inches(14,6)
x = [x for x in range (27,45)];
g, = plt.plot(x, sum27, 'g^', color=color[1], markersize = 15)
for j in range(len(sum27)):                                         
    plt.annotate(str(round(sum27[j],3)), xy=(x[j]+0.001,sum27[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig27_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 27-44', fontsize = 20)
plt.xlim(27,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig27_rri.savefig('rri_7.0_27_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 28) """

GFR1_28_ = [34.11, 21.13, 23.98, 15.96, 12.42, 5.46, 8.32, 8.62, 11.56, 
           7.44, 4.57, 1.54, 3.33, 1.97, 0.81, 0.57,
            0.09, 0.59, 0.33, 0.22, 0.2, 0.11]
GFR2_28_ = [27.96, 27.5, 30, 28.08, 28.08, 22.9, 20.03, 18.7, 12.57, 11.51,
           8.11, 4.26, 6.05, 2.54, 2.94, 1.14, 
           1.04, 0.73, 0.43, 0.67, 0.51, 0.42]
GFR3_28_ = [0.00, 5.28, 4.6, 3.97, 2.5, 3.76, 1.56, 1.95, 2.18, 2.37, 2.76,
           0.88, 1.73, 0.6, 1.12, 0.76, 0.57, 0.45, 
           0.65, 0.33, 0.2, 0.11]

fig28_gfr = plt.gcf()
fig28_gfr.set_size_inches(8,6)
x = [x for x in range (28, 50)];
plt.xlim(28,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_28_, color=color[3])
g, = plt.plot(x, GFR2_28_, color=color[2])
g, = plt.plot(x, GFR3_28_, color=color[1])
fig28_gfr.suptitle('GFR in 2000-2014 at different ages (28-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig28_gfr.savefig('gfr00_7.0_28_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr28_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr28_ = []

for i in range (len(GFR1_28_)):
    pr28_.append(GFR1_28_[i]/1000)
    pr28_.append(GFR2_28_[i]/1000)
    pr28_.append(GFR3_28_[i]/1000)
    pr28_s.append(pr28_)
    pr28_ = []

frpv_28 = deepcopy(completeSimulation(pr28_s, pb00s_1, pb00s_2, pb00s_3, 17))

fig28_rps = plt.gcf()
fig28_rps.set_size_inches(11,6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph28 = []
for i in range (len(frpv_28)):
    g, = plt.plot(x, frpv_28[i], 'bs', color=color[i], markersize = 10) 
#    if i == len(frpv_24) - 1:
#        for j in range (len(frpv_22[i])):                                   
#            plt.annotate(str(round(frpv_22[i][j],5)), xy=(x[j]+0.1,frpv_22[i][j]+0.01*x[j]),
#                horizontalalignment='left',verticalalignment='bottom') 
    cgraph28.append(g)
fig28_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 28-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph28], ['28', '29', '30', '31', '32', 
'33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44'], 
       loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig28_rps.savefig('rps00_7.0_28_China20s.png', dpi=100)

sum28 = []
for i in frpv_28:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum28.append(rri_sum)

fig28_rri = plt.gcf()
fig28_rri.set_size_inches(14,6)
x = [x for x in range (28,45)];
g, = plt.plot(x, sum28, 'g^', color=color[1], markersize = 15)
for j in range(len(sum28)):                                         
    plt.annotate(str(round(sum28[j],3)), xy=(x[j]+0.001,sum28[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig28_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 28-44', fontsize = 20)
plt.xlim(28,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig28_rri.savefig('rri_7.0_28_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 29) """

GFR1_29_ = [21.13, 13.27, 15.96, 9.81, 8.5, 4.19, 7.5, 9.34, 7.72, 5.47, 3.67,
           1.73, 1.14, 1.21, 0.57, 0.09, 0.59, 0.33, 0.22, 0.2, 0.11]
GFR2_29_ = [27.5, 26.4, 28.08, 24.13, 20.95, 17.38, 15.42, 13.71, 10.07, 7.58, 
           6.03, 2.3, 3.43, 2.14, 1.14, 1.04, 0.73, 0.43, 0.67, 0.51, 0.42]
GFR3_29_ = [0.00, 5.03, 3.97, 3.31, 2.28, 3.3, 3.14, 1.92, 1.94, 2.31, 2.18, 
           1.17, 1.23, 0.52, 0.76, 0.57, 0.45, 0.65, 0.33, 0.2, 0.11]

fig29_gfr = plt.gcf()
fig29_gfr.set_size_inches(8,6)
x = [x for x in range (29, 50)];
plt.xlim(29,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_29_, color=color[3])
g, = plt.plot(x, GFR2_29_, color=color[2])
g, = plt.plot(x, GFR3_29_, color=color[1])
fig29_gfr.suptitle('GFR in 2000-2014 at different ages (29-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig29_gfr.savefig('gfr00_7.0_29_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr29_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr29_ = []

for i in range (len(GFR1_29_)):
    pr29_.append(GFR1_29_[i]/1000)
    pr29_.append(GFR2_29_[i]/1000)
    pr29_.append(GFR3_29_[i]/1000)
    pr29_s.append(pr29_)
    pr29_ = []

frpv_29 = deepcopy(completeSimulation(pr29_s, pb00s_1, pb00s_2, pb00s_3, 16))

fig29_rps = plt.gcf()
fig29_rps.set_size_inches(11,6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph29 = []
for i in range (len(frpv_29)):
    g, = plt.plot(x, frpv_29[i], 'bs', color=color[i], markersize = 10) 
#    if i == len(frpv_24) - 1:
#        for j in range (len(frpv_22[i])):                                   
#            plt.annotate(str(round(frpv_22[i][j],5)), xy=(x[j]+0.1,frpv_22[i][j]+0.01*x[j]),
#                horizontalalignment='left',verticalalignment='bottom') 
    cgraph29.append(g)
fig29_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 29-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph29], ['29', '30', '31', '32', 
'33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44'], 
       loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig29_rps.savefig('rps00_7.0_29_China20s.png', dpi=100)

sum29 = []
for i in frpv_29:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum29.append(rri_sum)

fig29_rri = plt.gcf()
fig29_rri.set_size_inches(14,6)
x = [x for x in range (29,45)];
g, = plt.plot(x, sum29, 'g^', color=color[1], markersize = 15)
for j in range(len(sum29)):                                         
    plt.annotate(str(round(sum29[j],3)), xy=(x[j]+0.001,sum29[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig29_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 29-44', fontsize = 20)
plt.xlim(29,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig29_rri.savefig('rri_7.0_29_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 30) """

GFR1_30_ = [13.27, 7.99, 9.81, 6.58, 6.43, 2.66, 5.01, 5.95, 8.75, 6.18, 3.55,
           1.58, 1.7, 1.06, 0.09, 0.59, 0.33, 0.22, 0.2, 0.11]
GFR2_30_ = [26.4, 24.07, 24.13, 19.16, 16.95, 12.92, 11.16, 12.58, 9.9, 5.5,
           5.12, 2.81, 1.95, 2.19, 1.04, 0.73, 0.43, 0.67, 0.51, 0.42]
GFR3_30_ = [0.00, 4.57, 3.31, 3.47, 2.01, 2.68, 2.53, 2.46, 1.26, 1.51, 2.14, 
           0.59, 1.49, 1.01, 0.57, 0.45, 0.65, 0.33, 0.2, 0.11]

fig30_gfr = plt.gcf()
fig30_gfr.set_size_inches(8,6)
x = [x for x in range (30, 50)];
plt.xlim(30,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_30_, ls = "solid", color=color[3])
g, = plt.plot(x, GFR2_30_, color=color[2])
g, = plt.plot(x, GFR3_30_, color=color[1])
fig30_gfr.suptitle('GFR in 2000-2014 at different ages (30-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig30_gfr.savefig('gfr00_7.0_30_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr30_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr30_ = []

for i in range (len(GFR1_30_)):
    pr30_.append(GFR1_30_[i]/1000)
    pr30_.append(GFR2_30_[i]/1000)
    pr30_.append(GFR3_30_[i]/1000)
    pr30_s.append(pr30_)
    pr30_ = []

frpv_30 = deepcopy(completeSimulation(pr30_s, pb00s_1, pb00s_2, pb00s_3, 15))

fig30_rps = plt.gcf()
fig30_rps.set_size_inches(11, 6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph30 = []
for i in range (len(frpv_30)):
    g, = plt.plot(x, frpv_30[i], 'bs', color=color[i], markersize = 10) 
#    if i == len(frpv_24) - 1:
#        for j in range (len(frpv_22[i])):                                   
#            plt.annotate(str(round(frpv_22[i][j],5)), xy=(x[j]+0.1,frpv_22[i][j]+0.01*x[j]),
#                horizontalalignment='left',verticalalignment='bottom') 
    cgraph30.append(g)
fig30_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 30-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph30], ['30', '31', '32', 
           '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44'], 
            loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig30_rps.savefig('rps00_7.0_30_China20s.png', dpi=100)

sum30 = []
for i in frpv_30:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum30.append(rri_sum)

fig30_rri = plt.gcf()
fig30_rri.set_size_inches(14,6)
x = [x for x in range (30,45)];
g, = plt.plot(x, sum30, 'g^', color=color[1], markersize = 15)
for j in range(len(sum30)):                                         
    plt.annotate(str(round(sum30[j],3)), xy=(x[j]+0.001,sum30[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig30_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 30-44', fontsize = 20)
plt.xlim(30,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig30_rri.savefig('rri_7.0_30_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 31) """

GFR1_31_ = [7.99, 5.37, 6.58, 4.49, 4.3, 2, 3.68, 5.42, 7.56, 5.79, 2.64,
            0.76, 0.61, 0.86, 0.59, 0.33, 0.22, 0.2, 0.11]
GFR2_31_ = [24.07, 18.61, 19.16, 13.8, 11.65, 9.43, 10.1, 7.83, 6.38, 3.95, 3.5,
            1.33, 1.81, 1.38, 0.73, 0.43, 0.67, 0.51, 0.42]
GFR3_31_ = [0.00, 4.18, 3.47, 3.2, 1.46, 2.41, 1.39, 1.78, 1.26, 0.82, 1.52, 
            0.27, 0.54, 0.28, 0.45, 0.65, 0.33, 0.2, 0.11]

fig31_gfr = plt.gcf()
fig31_gfr.set_size_inches(8,6)
x = [x for x in range (31, 50)];
plt.xlim(31,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_31_, ls = "solid", color=color[3])
g, = plt.plot(x, GFR2_31_, color=color[2])
g, = plt.plot(x, GFR3_31_, color=color[1])
fig31_gfr.suptitle('GFR in 2000-2014 at different ages (31-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig31_gfr.savefig('gfr00_7.0_31_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr31_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr31_ = []

for i in range (len(GFR1_31_)):
    pr31_.append(GFR1_31_[i]/1000)
    pr31_.append(GFR2_31_[i]/1000)
    pr31_.append(GFR3_31_[i]/1000)
    pr31_s.append(pr31_)
    pr31_ = []

frpv_31 = deepcopy(completeSimulation(pr31_s, pb00s_1, pb00s_2, pb00s_3, 14))

fig31_rps = plt.gcf()
fig31_rps.set_size_inches(11, 6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph31 = []
for i in range (len(frpv_31)):
    g, = plt.plot(x, frpv_31[i], 'bs', color=color[i], markersize = 10) 
#    if i == len(frpv_24) - 1:
#        for j in range (len(frpv_22[i])):                                   
#            plt.annotate(str(round(frpv_22[i][j],5)), xy=(x[j]+0.1,frpv_22[i][j]+0.01*x[j]),
#                horizontalalignment='left',verticalalignment='bottom') 
    cgraph31.append(g)
fig31_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 31-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph31], ['31', '32', 
           '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44'], 
            loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig31_rps.savefig('rps00_7.0_31_China20s.png', dpi=100)

sum31 = []
for i in frpv_31:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum31.append(rri_sum)

fig31_rri = plt.gcf()
fig31_rri.set_size_inches(14,6)
x = [x for x in range (31,45)];
g, = plt.plot(x, sum31, 'g^', color=color[1], markersize = 15)
for j in range(len(sum31)):                                         
    plt.annotate(str(round(sum31[j],3)), xy=(x[j]+0.001,sum31[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig31_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 31-44', fontsize = 20)
plt.xlim(31,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig31_rri.savefig('rri_7.0_31_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 32) """
#
GFR1_32_ = [5.37, 3.39, 4.49, 3.81, 2.83, 1.35, 2.89, 5.84, 5.81, 5.99, 2.9,
            1.11, 0.82, 1.21, 0.33, 0.22, 0.2, 0.11]
GFR2_32_ = [18.61, 11.69, 13.8, 10.42, 8.37, 6.76, 7.25, 6.75, 5.55, 2.83, 
            3.34, 0.97, 1.19, 0.9, 0.43, 0.67, 0.51, 0.42]
GFR3_32_ = [0.00, 3.5, 3.2, 2.37, 1.56, 1.99, 1.54, 0.67, 0.93, 0.76, 1.62, 
            0.22, 0.41, 0.78, 0.65, 0.33, 0.2, 0.11]

fig32_gfr = plt.gcf()
fig32_gfr.set_size_inches(8,6)
x = [x for x in range (32, 50)];
plt.xlim(32,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_32_, ls = "solid", color=color[3])
g, = plt.plot(x, GFR2_32_, color=color[2])
g, = plt.plot(x, GFR3_32_, color=color[1])
fig32_gfr.suptitle('GFR in 2000-2014 at different ages (32-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig32_gfr.savefig('gfr00_7.0_32_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr32_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr32_ = []

for i in range (len(GFR1_32_)):
    pr32_.append(GFR1_32_[i]/1000)
    pr32_.append(GFR2_32_[i]/1000)
    pr32_.append(GFR3_32_[i]/1000)
    pr32_s.append(pr32_)
    pr32_ = []

frpv_32 = deepcopy(completeSimulation(pr32_s, pb00s_1, pb00s_2, pb00s_3, 13))

fig32_rps = plt.gcf()
fig32_rps.set_size_inches(11, 6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph32 = []
for i in range (len(frpv_32)):
    g, = plt.plot(x, frpv_32[i], 'bs', color=color[i], markersize = 10) 
#    if i == len(frpv_24) - 1:
#        for j in range (len(frpv_22[i])):                                   
#            plt.annotate(str(round(frpv_22[i][j],5)), xy=(x[j]+0.1,frpv_22[i][j]+0.01*x[j]),
#                horizontalalignment='left',verticalalignment='bottom') 
    cgraph32.append(g)
fig32_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 32-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph32], ['32', 
           '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44'], 
            loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig32_rps.savefig('rps00_7.0_32_China20s.png', dpi=100)

sum32 = []
for i in frpv_32:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum32.append(rri_sum)

fig32_rri = plt.gcf()
fig32_rri.set_size_inches(14,6)
x = [x for x in range (32,45)];
g, = plt.plot(x, sum32, 'g^', color=color[1], markersize = 15)
for j in range(len(sum32)):                                         
    plt.annotate(str(round(sum32[j],3)), xy=(x[j]+0.001,sum32[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig32_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 32-44', fontsize = 20)
plt.xlim(32,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig32_rri.savefig('rri_7.0_32_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 33) """
#
GFR1_33_ = [3.39, 2.46, 3.81, 2.32, 1.67, 0.82, 2.54, 4.9, 6.15, 3.54, 2.17,
            0.38, 0.82, 0.56, 0.22, 0.2, 0.11]
GFR2_33_ = [11.69, 7.81, 10.42, 6.49, 5.92, 4.51, 5.52, 4.28, 3.8, 1.83, 2.42,
           0.95, 0.36, 0.83, 0.67, 0.51, 0.42]
GFR3_33_ = [0.00, 3, 2.37, 1.77, 1.51, 1.71, 0.97, 0.45, 0.95, 0.7, 1.14, 0.7,
            0.32, 0.5, 0.33, 0.2, 0.11]

fig_gfr = plt.gcf()
fig_gfr.set_size_inches(8,6)
x = [x for x in range (33, 50)];
plt.xlim(33,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_33_, ls = "solid", color=color[3])
g, = plt.plot(x, GFR2_33_, color=color[2])
g, = plt.plot(x, GFR3_33_, color=color[1])
fig_gfr.suptitle('GFR in 2000-2014 at different ages (33-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig_gfr.savefig('gfr00_7.0_33_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr33_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr33_ = []

for i in range (len(GFR1_33_)):
    pr33_.append(GFR1_33_[i]/1000)
    pr33_.append(GFR2_33_[i]/1000)
    pr33_.append(GFR3_33_[i]/1000)
    pr33_s.append(pr33_)
    pr33_ = []

frpv_33 = deepcopy(completeSimulation(pr33_s, pb00s_1, pb00s_2, pb00s_3, 12))

fig_rps = plt.gcf()
fig_rps.set_size_inches(11, 6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph = []
for i in range (len(frpv_33)):
    g, = plt.plot(x, frpv_33[i], 'bs', color=color[i], markersize = 10) 
#    if i == len(frpv_24) - 1:
#        for j in range (len(frpv_22[i])):                                   
#            plt.annotate(str(round(frpv_22[i][j],5)), xy=(x[j]+0.1,frpv_22[i][j]+0.01*x[j]),
#                horizontalalignment='left',verticalalignment='bottom') 
    cgraph.append(g)
fig_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 33-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph], ['33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44'], 
            loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig_rps.savefig('rps00_7.0_33_China20s.png', dpi=100)

sum = []
for i in frpv_33:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum.append(rri_sum)

fig_rri = plt.gcf()
fig_rri.set_size_inches(14,6)
x = [x for x in range (33,45)];
g, = plt.plot(x, sum, 'g^', color=color[1], markersize = 15)
for j in range(len(sum)):                                         
    plt.annotate(str(round(sum[j],3)), xy=(x[j]+0.001,sum[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 33-44', fontsize = 20)
plt.xlim(33,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig_rri.savefig('rri_7.0_33_China20s.png', dpi=100)

""" Fertility rate data in the 2000s (women start to bear children at 34) """
#
GFR1_34_ = [2.46, 1.85, 2.32, 1.17, 1.41, 0.68, 1.92, 3.58, 5.81, 4.12, 2.02,
            0.59, 0.94, 0.53, 0.2, 0.11]
GFR2_34_ = [7.81, 5.33, 6.49, 4.68, 3.41, 2.9, 3.36, 3.68, 2.01, 1.43, 2.04,
             0.71, 0.98, 1.06, 0.51, 0.42]
GFR3_34_ = [0.00, 2.47, 1.77, 1.57, 1.25, 1.36, 0.6, 0.33, 0.78, 0.69, 1.04,
            0.07, 0.31, 0.16, 0.2, 0.11]

fig_gfr = plt.gcf()
fig_gfr.set_size_inches(8,6)
x = [x for x in range (34, 50)];
plt.xlim(34,50)
plt.ylim(0, 140)
g, = plt.plot(x, GFR1_34_, ls = "solid", color=color[3])
g, = plt.plot(x, GFR2_34_, color=color[2])
g, = plt.plot(x, GFR3_34_, color=color[1])
fig_gfr.suptitle('GFR in 2000-2014 at different ages (34-year-old cohort)', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 24)
plt.ylabel('General Fertility Rate', fontsize = 24)
plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
fig_gfr.savefig('gfr00_7.0_34_China20s.png', dpi=100)

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr34_s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr34_ = []

for i in range (len(GFR1_34_)):
    pr34_.append(GFR1_34_[i]/1000)
    pr34_.append(GFR2_34_[i]/1000)
    pr34_.append(GFR3_34_[i]/1000)
    pr34_s.append(pr34_)
    pr34_ = []

frpv_34 = deepcopy(completeSimulation(pr34_s, pb00s_1, pb00s_2, pb00s_3, 11))

fig_rps = plt.gcf()
fig_rps.set_size_inches(11, 6)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph = []
for i in range (len(frpv_34)):
    g, = plt.plot(x, frpv_34[i], 'bs', color=color[i], markersize = 10) 
#    if i == len(frpv_24) - 1:
#        for j in range (len(frpv_22[i])):                                   
#            plt.annotate(str(round(frpv_22[i][j],5)), xy=(x[j]+0.1,frpv_22[i][j]+0.01*x[j]),
#                horizontalalignment='left',verticalalignment='bottom') 
    cgraph.append(g)
fig_rps.suptitle('Reproductive Probability Vectors of Women \n Starting Childbearing from 34-44', fontsize = 16)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 24)
plt.ylabel('Probabilities in Each State', fontsize = 24)
plt.legend([i for i in cgraph], ['34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44'], 
            loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 8)
plt.grid()
plt.show()
fig_rps.savefig('rps00_7.0_34_China20s.png', dpi=100)

sum = []
for i in frpv_34:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum.append(rri_sum)

fig_rri = plt.gcf()
fig_rri.set_size_inches(14,6)
x = [x for x in range (34,45)];
g, = plt.plot(x, sum, 'g^', color=color[1], markersize = 15)
for j in range(len(sum)):                                         
    plt.annotate(str(round(sum[j],3)), xy=(x[j]+0.001,sum[j]+0.001),
                     horizontalalignment='left',verticalalignment='bottom')
fig_rri.suptitle('Relative Risk Indexes of Women Starting Childbearing from 34-44', fontsize = 20)
plt.xlim(34,45)
plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
plt.yticks(np.arange(0.86, 0.90, 0.01))
plt.grid()
plt.xlabel('Age at first birth', fontsize = 24)
plt.ylabel('Relative Risk Index', fontsize = 24)
plt.show()
fig_rri.savefig('rri_7.0_34_China20s.png', dpi=100)

























