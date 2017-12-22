import numpy as np
from matplotlib import pyplot
A = np.array([[56.0, 0.0 ,4.4, 68.0],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])

print(A)

cal = A.sum(axis=0)

print(cal)

per = 100 * A / cal.reshape(1, 4)
print(per)

a = np.random.rand(4,5)
b = np.ones((5,2)) * 100

# a = np.matrix(a)
# b = np.matrix(b)
c = np.array([[1,2,3,4,5]])

print(a.shape)
print(b.shape)

a = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11]])
print(a)
a[:,[1]] = a[:, [1]] + 10
a = np.random.randn(5,400000)
n = a.size
a = np.reshape(a, (n // 2, 2))
a[:,[0]] = a[:,[0]] + 10;
print(a[:,[0]].shape)
pyplot.hist(a,100,ec='black');
pyplot.show()

import numpy as np
from matplotlib import pyplot
k = 1000000
a = np.random.rand(k) * 10
a = a ** 2;
pyplot.hist(a, 100, ec='black')
pyplot.show()

pyplot.hist()

pyplot.hist()