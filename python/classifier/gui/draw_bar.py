import os
import numpy as np
from matplotlib import pyplot as plt



l1 = []
l2 = []
l3 = []
with open(r"c:\users\deng\desktop\befor.txt", "r", encoding="utf-8") as f:
    x = f.readlines()
for a in x:
    l1.append(int(a.split()[0]))
with open(r"c:\users\deng\desktop\af.txt", "r", encoding="utf-8") as f:
    x = f.readlines()
for a in x:
    t = int(a.split()[0])
    if t <= 60:
        t = 40
    l2.append(t)
# with open(r"c:\users\deng\desktop\ttt.txt", "r", encoding="utf-8") as f:
#     x = f.readlines()
# for a in x:
#     t = int(a.split(" ")[-1])
#     l3.append(t)
plt.figure()
plt.bar(np.arange(len(l1)), l1, facecolor="blue", label="before")
axes = plt.gca()
axes.set_ylim([0, 1000])
plt.figure()
plt.bar(np.arange(len(l2)), l2, width=1, facecolor="red", label="after")
# plt.figure()
# plt.bar(np.arange(len(l3)), l3,label="after")
axes = plt.gca()
axes.set_ylim([0, 1000])
#plt.legend(loc="upper left")
plt.show()
