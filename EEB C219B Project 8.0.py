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
rs = matrix([[0,0.2643,0,0,0.7357,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
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
#            if (i == 0):
#                if (j == 1):
#                    tran[i,j] = pr[0] * pb1[3]
#                elif (j == 4):
#                    tran[i,j] = pr[0] * (1 - pb1[3])
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
#                elif j == i + 3:
#                    tran[i,j] = (1 - (1 - pr[1]) * (pb1[3])) * pb1[0]*(1-pr[1])
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
        if (len(pr) > 1):
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
        PCN.append(GFR[i]/1000)
    sum_ = sum(PCN)
    PCN.append(1-sum_)
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
GFR2_20 =  [2.79, 5.57, 8.86, 14.12, 17.62, 22.102, 26.29, 28.75, 31.85,	31.60,
            30.15, 29.328, 25.73, 20.938, 17.11, 13.13, 9.886, 7.92, 5.864,
            4.31, 3.17, 2.24, 1.46,	1.244, 0.95, 0.85, 0.79,	0.72, 0.62, 0.59]
GFR3_20 =  [0.0, 0.25, 0.81, 1.4,	2.22, 2.59, 3.37, 3.73, 4.138, 4.102, 4.494,	
            4.36, 3.94, 3.71, 3.18, 2.644,	2.306,	2.1,	1.88, 1.25,
            1.012, 0.80, 0.77,	0.55, 0.47, 0.41, 0.32, 0.29, 0.298, 0.31]
        
#fig00_gfr = plt.gcf()
#fig00_gfr.set_size_inches(10,8)
#x = [x for x in range (20, 50)];
#plt.ylim(0, 110, 10)
##plt.yticks(np.arange(0, 1.1, 0.1))
#g, = plt.plot(x, GFR1_20, color=color[3])
#g, = plt.plot(x, GFR2_20, color=color[2])
#g, = plt.plot(x, GFR3_20, color=color[1])
#fig00_gfr.suptitle('Average Age-Specific Fertility Rates (2000-2014)', fontsize = 24)
#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
#plt.xlabel('Ages', fontsize = 24)
#plt.ylabel('Age-Specific Fertility Rate', fontsize = 24)
#plt.legend(['First Birth', 'Second Birth', 'Third Birth'], 
#           loc='best')
#plt.grid()
#plt.show()
##fig00_rps.savefig('rps00s.pdf', dpi=100)
#fig00_gfr.savefig('gfr00_8.0_Average_China20s.png', dpi=100)
#
#""" Fertility rate data in the 2000s (women start to bear children at 20) """

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

frpv = deepcopy(completeSimulation(pr20s, pb00s_1, pb00s_2, pb00s_3, 30))

#fig00_rps = plt.gcf()
#fig00_rps.set_size_inches(14,7)
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
#fig00_rps.suptitle('Reproductive Probability Vectors of Women Starting Childbearing at Different Ages', fontsize = 20)
#labels_ = ["[0,0]","[1,0]", "[2,0]", "[3,0]",
#       "[1,<12]", "[2,<12]", "[3,<12]",
#       "[1,12-24]","[2,12-24]","[3,12-24]",
#       "[1,>24]", "[2,>24]", "[3,>24]"]
#plt.xticks(range(1,14), labels_, size = "small")
#plt.xlabel('Reproductive States', fontsize = 24)
#plt.ylabel('Probability in Each State', fontsize = 24)
#plt.legend([i for i in cgraph00], ['20', '21', '22', '23', '24', '25', 
#      '26', '27','28', '29', '30', '31', '32', '33', '34', '35', '36', '37',
#      '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49'], 
#       loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol = 10)
#plt.grid()
#plt.show()
##fig00_rps.savefig('rps00s.pdf', dpi=100)
#fig00_rps.savefig('rps00_8.0_Average_China20s.png', dpi=100)
##
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
#
#fig00_rri = plt.gcf()
#fig00_rri.set_size_inches(18,8)
#x = [x for x in range (20,50)];
#g, = plt.plot(x, sum_, 'ro', color=color[1], markersize = 15)
#for j in range(len(sum_)):                                         
#    plt.annotate(str(round(sum_[j],3)), xy=(x[j]-0.005,sum_[j]-0.002),
#                     horizontalalignment='left',verticalalignment='bottom')
#fig00_rri.suptitle('Relative Risk Index of Women Starting Childbearing Process at Different Ages', fontsize = 22)
#plt.xticks(np.arange(min(x)-1, max(x)+2, 1.0))
#plt.ylim(0.86, 0.90, 0.01)
#plt.grid()
#plt.xlabel('Age at first birth', fontsize = 20)
#plt.ylabel('Relative Risk Index', fontsize = 20)
#plt.show()
##fig00_rps.savefig('rps00s.pdf', dpi=100)
#fig00_rri.savefig('rri_8.0_Average_China20s.png', dpi=100)
##
#""" Probability of not having a child till X age.
#    Every entry represents the percentage of female population bearing child at
#    at a certain age. The last entry represents no children in whole life."""
PCN = haveChildAt(GFR1_20)
#
#"""match the individual to the whole populaiton of the average cohort"""
pop_frpv = np.zeros(13)
sum_frpv = 0
for i in range(15):
    for j in range (len(frpv[0])):
        pop_frpv[j] += PCN[i] * frpv[i][j]
pop_frpv[0] = PCN[(len(PCN)-1)]
#print(pop_frpv)

#fig00_p_rps = plt.gcf()
#fig00_p_rps.set_size_inches(10,6)
#x = [x for x in range (1,14)];
#plt.ylim(-0.1,1,0.1)
#plt.yticks(np.arange(0, 1.1, 0.1))
#cgraph00 = []
#g, = plt.plot(x, pop_frpv, 'bs', color=color[2], markersize = 10)
#for j in range(len(pop_frpv)):                                        
#    plt.annotate(str(round(pop_frpv[j],3)), xy=(x[j]+0.1,pop_frpv[j]+0.01*x[j]),
#                 horizontalalignment='left',verticalalignment='bottom')
#fig00_p_rps.suptitle('Fraction of the Cohort Population in each Reproductive State', fontsize = 18)
#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
#plt.xlabel('Reproductive States', fontsize = 24)
#plt.ylabel('Probabilities in Each State', fontsize = 24)
#plt.grid()
#plt.show()
##fig00_rps.savefig('rps00s.pdf', dpi=100)
#fig00_p_rps.savefig('rps00_8.0_Average_Pop_China20s.png', dpi=100)

GFR2_20_2 = deepcopy(GFR2_20)
for i in range(len(GFR2_20)):
    GFR2_20_2[i] *= 2
    
pr20s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr20 = []

for i in range (len(GFR1_20)):
    pr20.append(GFR1_20[i]/1000)
    pr20.append(GFR2_20_2[i]/1000)
    pr20.append(GFR3_20[i]/1000)
    pr20s.append(pr20)
    pr20 = []
#print (pr20s)

frpv_2 = deepcopy(completeSimulation(pr20s, pb00s_1, pb00s_2, pb00s_3, 30))

sum_2 = []
for i in frpv_2:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum_2.append(rri_sum)
    
pop_frpv_2 = np.zeros(13)
sum_frpv_2 = 0
for i in range(15):
    for j in range (len(frpv_2[0])):
        pop_frpv_2[j] += PCN[i] * frpv_2[i][j]
pop_frpv_2[0] = PCN[(len(PCN)-1)]
#
## ----------------------------------------------------------------------------#
GFR2_20_3 = deepcopy(GFR2_20)
for i in range(len(GFR2_20)):
    GFR2_20_3[i] *= 3

""" Fertility rate data in the 2000s (women start to bear children at 20) """

# List of the transitional probability vector for parous states each year in 
# 2000-2014, for women start to bear children at 20 in 2000
pr20s = []
# Each year's transitional probability vector for parous states, ready for implementation
pr20 = []

for i in range (len(GFR1_20)):
    pr20.append(GFR1_20[i]/1000)
    pr20.append(GFR2_20_3[i]/1000)
    pr20.append(GFR3_20[i]/1000)
    pr20s.append(pr20)
    pr20 = []
#print (pr20s)

frpv_3 = deepcopy(completeSimulation(pr20s, pb00s_1, pb00s_2, pb00s_3, 30))

"""match the individual to the whole populaiton of the average cohort"""
pop_frpv_3 = np.zeros(13)
sum_frpv_3 = 0
for i in range(15):
    for j in range (len(frpv_3[0])):
        pop_frpv_3[j] += PCN[i] * frpv_3[i][j]
pop_frpv_3[0] = PCN[(len(PCN)-1)]
#print(pop_frpv)

sum_3 = []
for i in frpv_3:
    rri_sum = 0
    for j in range (len(i)):
        rri_sum += i[j]*risk_index[j]
    sum_3.append(rri_sum)

#
""" For each year in 2000s, based on the women's reproductive state, we are
    gonna use the relative risk index to calculate their relative risk of 
    developing breast cancer
"""
fig00_gfr = plt.gcf()
fig00_gfr.set_size_inches(12,7)
x = [x for x in range (20, 50)];
plt.ylim(0, 130, 10)
#plt.yticks(np.arange(0, 1.1, 0.1))
g, = plt.plot(x, GFR1_20, color='r', linewidth=2)
g, = plt.plot(x, GFR2_20, color='blue', linewidth=2)
g, = plt.plot(x, GFR2_20_2, color='cyan', linewidth=2)
g, = plt.plot(x, GFR2_20_3, color='indigo', linewidth=2)
g, = plt.plot(x, GFR3_20, color='green', linewidth=2)
fig00_gfr.suptitle('Age-Specific Fertility Rate (2000-2014)', fontsize = 22)
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Ages', fontsize = 20)
plt.ylabel('Age-Specific Fertility Rate', fontsize = 20)
plt.legend(['First Birth', 'Second Birth(Reality)', 'Second Birth(2X)', 'Second Birth(3X)', 'Third Birth'], 
           loc='best')
plt.grid()
plt.show()
#fig00_rps.savefig('rps00s.pdf', dpi=100)
fig00_gfr.savefig('gfr00_8.0_Average_2x20s.png', dpi=100)
#
##print (sum_)
##
fig00_rri = plt.gcf()
fig00_rri.set_size_inches(12,7)
x = [x for x in range (20,50)];
g, = plt.plot(x, sum_3, 'ro', color=color[2], markersize = 10)
g, = plt.plot(x, sum_2, 'ro', color=color[3], markersize = 10)
g, = plt.plot(x, sum_, 'ro', color=color[1], markersize = 10)
#for j in range(len(sum_3)):                                         
#    plt.annotate(str(round(sum_3[j],3)), xy=(x[j]-0.005,sum_3[j]+0.001),
#                     horizontalalignment='left',verticalalignment='bottom')
fig00_rri.suptitle('Relative Risk Index in Cohorts with Different Age-Specific Fertility Rates', fontsize = 22)
plt.xticks(np.arange(min(x)-1, max(x)+1, 1.0))
plt.ylim(0.83, 0.90, 0.02)
plt.grid()
plt.xlabel('Age at First Birth', fontsize = 20)
plt.ylabel('Relative Risk Index', fontsize = 20)
plt.legend(['2nd Birth 3X', '2nd Birth 2X', 'Reality'], loc='best')
plt.show()
#fig00_rps.savefig('rps00s.pdf', dpi=100)
fig00_rri.savefig('rri_8.0_Average_3x20s.png', dpi=100)

fig00_p_rps = plt.gcf()
fig00_p_rps.set_size_inches(12,7)
x = [x for x in range (1,14)];
plt.ylim(-0.1,1,0.1)
plt.yticks(np.arange(0, 1.1, 0.1))
cgraph00 = []
g, = plt.plot(x, pop_frpv_3, 'bs', color=color[1], markersize = 10)
g, = plt.plot(x, pop_frpv_2, 'bs', color=color[3], markersize = 10)
g, = plt.plot(x, pop_frpv, 'bs', color=color[2], markersize = 10)
#for j in range(len(pop_frpv_3)):                                        
#    plt.annotate(str(round(pop_frpv_3[j],3)), xy=(x[j]+0.1,pop_frpv_3[j]+0.01*x[j]),
#                 horizontalalignment='left',verticalalignment='bottom')
fig00_p_rps.suptitle('Fraction of the Cohort Population in Each Reproductive State', fontsize = 22)
labels_ = ["[0,0]","[1,0]", "[2,0]", "[3,0]",
       "[1,<12]", "[2,<12]", "[3,<12]",
       "[1,12-24]","[2,12-24]","[3,12-24]",
       "[1,>24]", "[2,>24]", "[3,>24]"]
plt.xticks(range(1,14), labels_, size = "small")
#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.xlabel('Reproductive States', fontsize = 20)
plt.ylabel('Fraction of Cohort in Each State', fontsize = 20)
plt.legend(['2nd Birth 3X', '2nd Birth 2X', 'Reality'], loc='best')
plt.grid()
plt.show()
#fig00_rps.savefig('rps00s.pdf', dpi=100)
fig00_p_rps.savefig('rps00_8.0_Average_Pop_3x20s.png', dpi=100)

"""------------------------------------------------------------------------"""

#GFR1_20 =  [42.93, 75.63, 94.76, 105.84, 105.60, 92.57, 78.06, 62.44, 48.73, 
#            37.25, 25.85, 19.50, 14.01, 10.35, 8.08, 6.09, 4.88, 3.64, 3.09,
#            2.66, 2.23, 1.94, 1.582, 1.47, 1.32, 1.26, 1.19, 1.268, 1.219, 1.2]
#GFR2_20 =  [2.79, 5.57, 8.86, 14.12, 17.62, 22.102, 26.29, 28.75, 31.85,	31.60,
#            30.15, 29.328, 25.73, 20.938, 17.11, 13.13, 9.886, 7.92, 5.864,
#            4.31, 3.17, 2.24, 1.46,	1.244, 0.95, 0.85, 0.79,	0.72, 0.62, 0.59]
#GFR3_20 =  [0.0, 0.25, 0.81, 1.4,	2.22, 2.59, 3.37, 3.73, 4.138, 4.102, 4.494,	
#            4.36, 3.94, 3.71, 3.18, 2.644,	2.306,	2.1,	1.88, 1.25,
#            1.012, 0.80, 0.77,	0.55, 0.47, 0.41, 0.32, 0.29, 0.298, 0.31]
            
#GFR1_20_D1 =  [20, 32, 65, 80,	105.84, 105.60, 92.57, 78.06, 62.44, 48.73, 
#               37.25, 25.85, 19.50, 14.01, 10.35, 8.08, 6.09, 4.88, 3.64, 3.09,
#               2.66, 2.23, 1.94, 1.582, 1.47, 1.32, 1.26, 1.19, 1.268, 1.219]
#GFR2_20_D1 =  [1.24, 1.46, 2.79, 5.57, 8.86, 14.12, 17.62, 22.102, 26.29, 28.75, 31.85, 31.60,
#              30.15, 29.328, 25.73, 20.938, 17.11, 13.13, 9.886, 7.92, 5.864,
#              4.31, 3.17, 2.24, 1.46,	1.244, 0.95, 0.85, 0.79,	0.72]
#GFR3_20_D1 =  [0.0, 0.25, 0.32, 0.47, 0.81, 1.4, 2.22, 2.59, 3.37, 3.73, 4.138, 
#               4.102, 4.494,	4.36, 3.94, 3.71, 3.18, 2.644,	2.306,	2.1,	1.88, 1.25,
#               1.012, 0.80, 0.77,	0.55, 0.47, 0.41, 0.32, 0.29]
#               
#GFR1_20_D2 =  [15.85, 20, 23, 25, 30, 35, 50,	95.84, 95.60, 78.06, 62.44, 48.73, 
#               37.25, 25.85, 19.50, 14.01, 10.35, 8.08, 6.09, 4.88, 3.64, 3.09,
#               2.66, 2.23, 1.94, 1.582, 1.47, 1.32, 1.26, 1.19]
#GFR2_20_D2 =  [1.24, 1.46, 2.3, 4.8, 6.5, 7.5, 10, 14.12, 17.62, 22.102, 26.29, 28.75, 31.85,	31.60,
#               30.15, 29.328, 25.73, 20.938, 17.11, 13.13, 9.886, 7.92, 5.864,
#               4.31, 3.17, 2.24, 1.46,	1.244, 0.95, 0.85]
#GFR3_20_D2 =  [0.0, 0.25, 0.32, 0.47, 0.55, 0.68, 0.81, 1.4, 2.22, 2.59, 3.37, 3.73, 4.138, 
#               4.102, 4.494,	4.36, 3.94, 3.71, 3.18, 2.644,	2.306,	2.1,	1.88, 1.25,
#               1.012, 0.80, 0.77,	0.55, 0.47, 0.41]

""" Fertility rate data in the 2000s (women start to bear children at 20) """

## List of the transitional probability vector for parous states each year in 
## 2000-2014, for women start to bear children at 20 in 2000
#pr20s = []
## Each year's transitional probability vector for parous states, ready for implementation
#pr20 = []
#
#for i in range (len(GFR1_20)):
#    pr20.append(GFR1_20_D1[i]/1000)
#    pr20.append(GFR2_20_D1[i]/1000)
#    pr20.append(GFR3_20_D1[i]/1000)
#    pr20s.append(pr20)
#    pr20 = []
#
#frpv_D1 = deepcopy(completeSimulation(pr20s, pb00s_1, pb00s_2, pb00s_3, 30))

"""match the individual to the whole populaiton of the average cohort"""
#pop_frpv_D1 = np.zeros(13)
#sum_frpv_D1 = 0
#for i in range(15):
#    for j in range (len(frpv_D1[0])):
#        pop_frpv_D1[j] += PCN[i] * frpv_D1[i][j]
#pop_frpv_D1[0] = PCN[(len(PCN)-1)]
#
#sum_D1 = []
#for i in frpv_D1:
#    rri_sum = 0
#    for j in range (len(i)):
#        rri_sum += i[j]*risk_index[j]
#    sum_D1.append(rri_sum)
#print(sum_)
#print(sum_D1)
#    
#pr20s = []
## Each year's transitional probability vector for parous states, ready for implementation
#pr20 = []
#
#for i in range (len(GFR1_20)):
#    pr20.append(GFR1_20_D2[i]/1000)
#    pr20.append(GFR2_20_D2[i]/1000)
#    pr20.append(GFR3_20_D2[i]/1000)
#    pr20s.append(pr20)
#    pr20 = []
#
#frpv_D2 = deepcopy(completeSimulation(pr20s, pb00s_1, pb00s_2, pb00s_3, 30))

"""match the individual to the whole populaiton of the average cohort"""
#pop_frpv_D2 = np.zeros(13)
#sum_frpv_D2 = 0
#for i in range(15):
#    for j in range (len(frpv_D2[0])):
#        pop_frpv_D2[j] += PCN[i] * frpv_D2[i][j]
#pop_frpv_D2[0] = PCN[(len(PCN)-1)]
#
#sum_D2 = []
#for i in frpv_D2:
#    rri_sum = 0
#    for j in range (len(i)):
#        rri_sum += i[j]*risk_index[j]
#    sum_D2.append(rri_sum)
#print (sum_D2)
    
#fig00_gfr = plt.gcf()
#fig00_gfr.set_size_inches(12,7)
#x = [x for x in range (20, 50)];
#plt.ylim(0, 130, 10)
##plt.yticks(np.arange(0, 1.1, 0.1))
#g, = plt.plot(x, GFR1_20, color='r', linewidth = 1.5)
#g, = plt.plot(x, GFR1_20_D1, color='gold', linewidth = 1.5)
#g, = plt.plot(x, GFR1_20_D2, color='indigo', linewidth = 1.5)
#g, = plt.plot(x, GFR2_20, color='blue', linewidth = 1.5)
#g, = plt.plot(x, GFR2_20_D1, color='green', linewidth = 1.5)
#g, = plt.plot(x, GFR2_20_D2, color='purple', linewidth = 1.5)
#g, = plt.plot(x, GFR3_20, color='brown', linewidth = 1.5)
#g, = plt.plot(x, GFR3_20_D1, color='gold', linewidth = 1.5)
#g, = plt.plot(x, GFR3_20_D2, color='black', linewidth = 1.5)
#fig00_gfr.suptitle('Age-Specific Fertility Rate (2000-2014)', fontsize = 24)
#plt.xticks(np.arange(min(x), max(x)+1, 1.0))
#plt.xlabel('Ages', fontsize = 24)
#plt.ylabel('General Fertility Rate', fontsize = 24)
#plt.legend(['1st Birth(Reality)', '2-year Delayed in 1st Birth','4-year Delay in 1st Birth', '2nd Birth(Reality)', 
#            '2-year Delay in 2nd Birth', '4-year Delay in 2nd Birth', '3rd Birth(Reality)', '2-year Delay in 3rd Birth',
#            '4-year Delay in 3rd Birth'], loc='best')
#plt.grid()
#plt.show()
##fig00_rps.savefig('rps00s.pdf', dpi=100)
#fig00_gfr.savefig('gfr00_8.0_Average_Delay_20s.png', dpi=100)
#
#fig00_rri = plt.gcf()
#fig00_rri.set_size_inches(12,7)
#x = [x for x in range (20,50)];
#g, = plt.plot(x, sum_D2, 'ro', color=color[2], markersize = 10)
#g, = plt.plot(x, sum_D1, 'ro', color=color[3], markersize = 10)
#g, = plt.plot(x, sum_, 'ro', color=color[1], markersize = 10)
##for j in range(len(sum_3)):                                         
##    plt.annotate(str(round(sum_3[j],3)), xy=(x[j]-0.005,sum_3[j]+0.001),
##                     horizontalalignment='left',verticalalignment='bottom')
#fig00_rri.suptitle('Relative Risk Index in Cohorts with Different Age-Specific Fertility Rates', fontsize = 20)
#plt.xticks(np.arange(min(x)-1, max(x)+1, 1.0))
#plt.ylim(0.83, 0.90, 0.02)
#plt.grid()
#plt.xlabel('Age at First Birth', fontsize = 24)
#plt.ylabel('Relative Risk Index', fontsize = 24)
#plt.legend(['4-year Delay', '2-year Delay', 'Reality'], loc='best')
#plt.show()
##fig00_rps.savefig('rps00s.pdf', dpi=100)
#fig00_rri.savefig('rri_8.0_Average_Delay_20s.png', dpi=100)
#
#fig00_p_rps = plt.gcf()
#fig00_p_rps.set_size_inches(12,7)
#x = [x for x in range (1,14)];
#plt.ylim(-0.1,1,0.1)
#plt.yticks(np.arange(0, 1.1, 0.1))
#cgraph00 = []
#g, = plt.plot(x, pop_frpv_D2, 'bs', color=color[1], markersize = 10)
#g, = plt.plot(x, pop_frpv_D1, 'bs', color=color[3], markersize = 10)
#g, = plt.plot(x, pop_frpv, 'bs', color=color[2], markersize = 10)
##for j in range(len(pop_frpv_3)):                                        
##    plt.annotate(str(round(pop_frpv_3[j],3)), xy=(x[j]+0.1,pop_frpv_3[j]+0.01*x[j]),
##                 horizontalalignment='left',verticalalignment='bottom')
#fig00_p_rps.suptitle('Fraction of the Cohort Population in each Reproductive State', fontsize = 18)
#labels_ = ["[0,0]","[1,0]", "[2,0]", "[3,0]",
#       "[1,<12]", "[2,<12]", "[3,<12]",
#       "[1,12-24]","[2,12-24]","[3,12-24]",
#       "[1,>24]", "[2,>24]", "[3,>24]"]
#plt.xticks(range(1,14), labels_, size = "small")
##plt.xticks(np.arange(min(x), max(x)+1, 1.0))
#plt.xlabel('Reproductive States', fontsize = 24)
#plt.ylabel('Fraction of Cohort in Each State', fontsize = 20)
#plt.legend(['4-year Delay', '2-year Delay', 'Reality'], loc='best')
#plt.grid()
#plt.show()
##fig00_rps.savefig('rps00s.pdf', dpi=100)
#fig00_p_rps.savefig('rps00_8.0_Average_Pop_Delay_20s.png', dpi=100)
#
#rri_1 = 0
#for i in range (13):
#    rri_1 += sum_[i]*pop_frpv[i]
#    
#rri_2 = 0
#for i in range (13):
#    rri_2 += sum_2[i]*pop_frpv_2[i]
#    
#rri_3 = 0
#for i in range (13):
#    rri_3 += sum_3[i]*pop_frpv_3[i]
#rri_pop = [rri_1, rri_2, rri_3]
#
#rri_D1 = 0
#for i in range (13):
#    rri_D1 += sum_D1[i]*pop_frpv[i]
#    
#rri_D2 = 0
#for i in range (13):
#    rri_D2 += sum_D2[i]*pop_frpv[i]
#
#rri_popD = [rri_1, rri_D1, rri_D2]

#fig00_gfr = plt.gcf()
#fig00_gfr.set_size_inches(12,7)
#x = [x for x in range (1, 4)];
#plt.ylim(0.80, 0.90, 0.02)
##plt.yticks(np.arange(0, 1.1, 0.1))
#g, = plt.plot(x, rri_pop, 'ro', color='r')
#g, = plt.plot(x, rri_popD, 'ro', color='blue')
#fig00_gfr.suptitle('Cohort Population Relative Risk Index', fontsize = 24)
#labels_ = ["","[0,0]","[1,0]", "[2,0]", ""]
#plt.xticks(range(0, 5), labels_, size = "small")
#plt.xlabel('Ages', fontsize = 24)
#plt.ylabel('General Fertility Rate', fontsize = 24)
#plt.legend(['Fertility Rate X', 'Delay'], 
#           loc='best')
#plt.grid()
#plt.show()
##fig00_rps.savefig('rps00s.pdf', dpi=100)
#fig00_gfr.savefig('gfr00_8.0_Average_rri_pop_X_20s.png', dpi=100)
















