import pandas as pd
import numpy as np
from geopy.distance import vincenty
import math
import scipy.optimize as opt
from scipy.optimize import minimize

df = pd.read_excel('/Users/alexandradinu/Desktop/CompEcon_Fall17/ProblemSets/radio_merger_data.xlsx')

"creating a scaled population column"
df["population_scaled"] = np.log(df["population_target"]/1000)

"creating a scaled price column"
df["price_scaled"] = np.log(df["price"]/1000)

"Part One"
"Setting up the data into an array for use in the score function"
"I first define the years and I create an empty array that will hold all"
"observations and counterfactuals. This code goes creates the obersevation by looping row by row."
"The obersvations are then placed in the result array which will be use to write score function. "
"The final array is 2421 X 12"

N1 = len(df[(df['year']<2008)])

N2 = len(df[(df['year']>2007)])

m = 1
N = N1
BT = 1

result_array = np.empty((0, 12))

while (m <= 2):
    while (BT <= N-1):
        K = 1
        while (K <= N-BT):
            point1 = (df.iloc[BT-1, 3], df.iloc[BT-1, 4])
            point2 = (df.iloc[BT-1, 5], df.iloc[BT-1, 6])
            point3 = (df.iloc[BT+K-1, 3], df.iloc[BT+K-1, 4])
            point4 = (df.iloc[BT+K-1, 5], df.iloc[BT+K-1, 6])

            x1bm_y1tm = df.iloc[BT-1, 9] * df.iloc[BT-1, 12]
            x2bm_y2tm = df.iloc[BT-1, 11] * df.iloc[BT-1, 12]
            dbtm = vincenty(point1, point2).miles

            x1qm_y1um = df.iloc[BT+K-1, 9] * df.iloc[BT+K-1, 12]
            x2qm_y1um = df.iloc[BT+K-1, 11] * df.iloc[BT+K-1, 12]
            dqu = vincenty(point3, point4).miles

            x1bm_y1um = df.iloc[BT-1, 9] * df.iloc[BT+K-1, 12]
            x2bm_y1um = df.iloc[BT-1, 11] * df.iloc[BT+K-1, 12]
            dbu = vincenty(point1, point4).miles

            x1qm_y1tm = df.iloc[BT+K-1, 9] * df.iloc[BT-1, 12]
            x2qm_y1tm = df.iloc[BT+K-1, 11] * df.iloc[BT-1, 12]
            dqt = vincenty(point3, point2).miles

            result = np.array([x1bm_y1tm, x2bm_y2tm, dbtm, x1qm_y1um, x2qm_y1um, dqu, x1bm_y1um, x2bm_y1um, dbu, x1qm_y1tm, x2qm_y1tm, dqt])
            K = K + 1
            result_array = np.append(result_array, [result], axis=0)

        BT = BT + 1
    N = N1 + N2 - 1

    m = m + 1

print(result_array)


"Creating the score function"
"Using the result array I calculate the score function. If the condition holds, the indicator function gets a value of -1"
"This allows me to maximize the score function through the minimization routine."

def mse(params, result_array):
    alpha, beta = params
    sum = 0
    i = 0
    while(i <= len(result_array)-1):
        fbt = result_array[i, 0] + alpha * result_array[i, 1] + beta * result_array[i, 2] + result_array[i, 3] + alpha * result_array[i, 4] + beta * result_array[i, 5] - result_array[i, 6] - alpha * result_array[i, 7] - beta * result_array[i, 8] - result_array[i, 9] - alpha * result_array[i, 10] - beta * result_array[i, 11]


        if fbt >= 0:
            sum = sum - 1

        i = i + 1
        print(sum)
    return sum

"Initial Guess"
b1 = (1, 1)

"Optimization routine"
f_b = opt.minimize(mse, b1, result_array, method = 'Nelder-Mead', options={'disp': True})

print(f_b)

"Part Two of the Homework:"
"Creating the second data array"
"Similar process although here I add prices and HHI in my result array for ease of use with the score function."

N1 = len(df[(df['year']<2008)])

N2 = len(df[(df['year']>2007)])

m = 1
N = N1
BT = 1

result_array2 = np.empty((0, 20))

while (m <= 2):
    while (BT <= N-1):
        K = 1
        while (K <= N-BT):
            point1 = (df.iloc[BT-1, 3], df.iloc[BT-1, 4])
            point2 = (df.iloc[BT-1, 5], df.iloc[BT-1, 6])
            point3 = (df.iloc[BT+K-1, 3], df.iloc[BT+K-1, 4])
            point4 = (df.iloc[BT+K-1, 5], df.iloc[BT+K-1, 6])

            x1bm_y1tm = df.iloc[BT-1, 9] * df.iloc[BT-1, 12]
            x2bm_y2tm = df.iloc[BT-1, 11] * df.iloc[BT-1, 12]
            HHIbtm = df.iloc[BT-1, 8]
            dbtm = vincenty(point1, point2).miles

            x1qm_y1um = df.iloc[BT+K-1, 9] * df.iloc[BT+K-1, 12]
            x2qm_y1um = df.iloc[BT+K-1, 11] * df.iloc[BT+K-1, 12]
            HHIum = df.iloc[BT+K-1, 8]
            dqu = vincenty(point3, point4).miles

            x1bm_y1um = df.iloc[BT-1, 9] * df.iloc[BT+K-1, 12]
            x2bm_y1um = df.iloc[BT-1, 11] * df.iloc[BT+K-1, 12]
            HHIbum = df.iloc[BT+K-1, 8]
            dbu = vincenty(point1, point4).miles

            x1qm_y1tm = df.iloc[BT+K-1, 9] * df.iloc[BT-1, 12]
            x2qm_y1tm = df.iloc[BT+K-1, 11] * df.iloc[BT-1, 12]
            HHImtm = df.iloc[BT-1, 8]
            dqt = vincenty(point3, point2).miles

            pbt = df.iloc[BT-1, 13]
            pum = df.iloc[BT+K-1, 13]

            result2 = np.array([x1bm_y1tm, x2bm_y2tm, HHIbtm, dbtm, x1bm_y1um, x2bm_y1um, HHIbum, dbu, pbt, pum, x1qm_y1um, x2qm_y1um, HHIum, dqu,  x1qm_y1tm, x2qm_y1tm, HHImtm, dqt, pum, pbt])
            K = K + 1
            result_array2 = np.append(result_array2, [result2], axis=0)

        BT = BT + 1
    N = N1 + N2 - 1

    m = m + 1

print(result_array2)

"Creating the score function"
def mse2(params2, result_array2):
    delta, alpha, gamma, beta = params2
    sum = 0
    i = 0
    while(i <= len(result_array2)-1):
        fbt1 = delta * result_array2[i, 0] + alpha * result_array2[i, 1] + gamma * result_array2[i, 2] + beta * result_array2[i, 3] - delta * result_array2[i, 4] - alpha * result_array2[i, 5] - gamma * result_array2[i, 6] - beta * result_array2[i, 7] - result_array2[i, 8] + result_array2[i, 9]

        fbt2 = delta * result_array2[i, 10] + alpha * result_array2[i, 11] + gamma * result_array2[i, 12] + beta * result_array2[i, 13] - delta * result_array2[i, 14] - alpha * result_array2[i, 15] - gamma * result_array2[i, 16] - beta * result_array2[i, 17] - result_array2[i, 18] + result_array2[i, 19]


        if fbt1 >= 0 and fbt2 >=0:
            sum = sum - 1

        i = i + 1
        print(fbt1, fbt2, sum)
    return sum

"Initial Guess"
b1 = (1, 2, -1, -1)

"Optimization routine"
f_b1 = opt.minimize(mse2, b1, result_array2, method = 'Nelder-Mead', options={'disp': True})

print(f_b1)
