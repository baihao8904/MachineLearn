# -*- coding:utf-8 -*-

import scipy as sp
import matplotlib.pyplot as plt

data = sp.genfromtxt("./data1/web_traffic.tsv",delimiter="\t")

x = data[:,0]
y = data[:,1]
print(sp.sum(sp.isnan(y)))
X = x[~sp.isnan(y)]
Y = y[~sp.isnan(y)]

plt.scatter(X,Y)
plt.title("web traffic ")
plt.xlabel("time")
plt.ylabel("hits/hour")
plt.xticks([w*7*24 for w in range(10)],["W %i" %w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
plt.show()