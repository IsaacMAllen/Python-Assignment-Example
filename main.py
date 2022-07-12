import numpy as np
import math


def gauss(A, b, n):
#
#   initialize index and scale vectors
#
  s = [0] * n
  l = [0] * n
  
    

#
#   make the scale vector the larger of the largest positive
#   number and smallest negative number in each row
#
  for i in range(0, n):
      l[i] = i
      smax = 0
      for j in range(0, n):
        if np.abs(A[i][j]) > smax:
          smax = np.abs(A[i][j])
      s[i] = smax
#
#   choose the largest ratio in column
#
  for k in range(0, n):
    j = k
    rmax = 0
    for i in range(k, n):
      r = np.abs(A[l[i]][k]/s[l[i]])
      
      if(r > rmax):
        rmax = r
        j = i
    temp = l[j]
    l[j] = l[k]
    l[k] = temp
#
#   perform the reduction
#
    
    for i in range(k + 1, n):
      xmult = A[l[i]][k]/A[l[k]][k]
      for j in range(k, n):
        A[l[i]][j] = A[l[i]][j] - (xmult * A[l[k]][j]) 
      b[l[i]] = b[l[i]]- xmult * b[l[k]]
#  
#   Now backsolve to find the solution
#
        

  x = [0] * n
  x[l[n-1]] = b[l[n-1]]/A[l[n-1]][n-1]
  for j in range(n-2, -1, -1):
    sum = 0
    for k in range(j+1, n):
      sum = sum + A[l[j]][k] * x[k]
    x[j] = (b[l[j]]-sum)/A[l[j]][j]
  return x
#
#   do the setup for Larger Numerical Example on page 87
#
x = gauss([[3, -13, 9, 3], [-6, 4, 1, -18], [6, -2, 2, 4], [12, -8, 6, 10]],[-19, -34, 16, 26], 4)
print("\nsolution to the Larger Numerical Example on page 87 \n" , x, "\n")
#
x = gauss([[2, 4, -2], [1, 3, 4],[5, 2, 0]],[6, -1, 2], 3)
print("\nsolution to the system given in #8 on page 99 \n" , x, "\n")

x = gauss([[2, 4, -2], [1, 3, 4],[5, 2, 0]],[6, -1, 2], 3)
print("\nsolution to the system given in #8 on page 99 \n" , x, "\n")

x = gauss([[1, 1/2, 1/3, 1/4, 1/5], [1/2, 1/3, 1/4, 1/5, 1/6],[1/3, 1/4, 1/5, 1/6, 1/7],[1/4, 1/5, 1/6, 1/7, 1/8],[1/5, 1/6, 1/7, 1/8, 1/9]], [1,0,0,0,0], 5)
print("\nsolution to the first system given in #19 on page 102" , x)


def residual(A, x, b):
  product = np.dot(A, x)
  return product - b
r = residual([[1, 1/2, 1/3, 1/4, 1/5], [1/2, 1/3, 1/4, 1/5, 1/6],[1/3, 1/4, 1/5, 1/6, 1/7],[1/4, 1/5, 1/6, 1/7, 1/8],[1/5, 1/6, 1/7, 1/8, 1/9]], x, [1,0,0,0,0])
print("Its residual:", r)

def l2norm(r):
  sum = 0
  for i in r:
    sum = sum + i**2
  return math.sqrt(sum)
print("Its residual's l2 norm:", l2norm(r))
error1 = np.abs(np.subtract(x, [25, -300, 1050, -1400, 630]))
print("The error vector:", error1)
print("Its error l2 norm:", l2norm(error1))
x = gauss([[1.0, 0.5, 0.333333, 0.25, 0.2], [0.5, 0.333333, 0.25, 0.2, 0.166667],[0.333333, 0.25, 0.2, 0.166667, 0.142857],[0.25, 0.2, 0.166667, 0.142857, 0.125],[0.2, 0.166667, 0.142857, 0.125, 0.111111]], [1,0,0,0,0], 5)
print("\nsolution to the second system given in #19 on page 102" , x)

r = residual([[1.0, 0.5, 0.333333, 0.25, 0.2], [0.5, 0.333333, 0.25, 0.2, 0.166667],[0.333333, 0.25, 0.2, 0.166667, 0.142857],[0.25, 0.2, 0.166667, 0.142857, 0.125],[0.2, 0.166667, 0.142857, 0.125, 0.111111]], x, [1,0,0,0,0])
print("Its residual:", r)
print("Its residual's l2 norm:", l2norm(r))
error2 = np.abs(np.subtract(x, [26.9314, -336.018, 1205.11, -1634.03, 744.411]))
print("And the error vector:", error2)
print("Its error l2 norm:", l2norm(error2))
print("\nThis indicates to me that the second setup of this system is less accurate than the first because the entries are rounded to six decimal places which cause the sensitive results to be much further from the exact answer than if it weren't.")