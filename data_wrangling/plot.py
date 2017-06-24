import numpy as np
import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import Scatter, Layout

x = [40,50,60,70,80,90,95] #remains constant 

 
y_b0=[0.3554090349346869, 0.37947842512972024, 0.39760426670509386, 0.4235646489357462, 0.43980275120545953, 0.35157091127161455, 0.3861882201447795]
y_b1=[0.3852703420915725, 0.4062982497049045, 0.42671715112498754, 0.45454196700090693, 0.47735461170417376, 0.3995775349435144, 0.43036796995789955]
y_b2=[0.6333654920772693, 0.6238474662640277, 0.6148941409777933, 0.6381719350640997, 0.6279728583909395, 0.6224014365544661, 0.6254475218557116]
y_b3=[0.6333654920772693, 0.6236258546720431, 0.6148941409777933, 0.6381719350640997, 0.6279728583909395, 0.6224014365544661, 0.6254475218557116]
y_b4=[0.633278248996316, 0.6236258546720431, 0.6148941409777933, 0.6381719350640997, 0.6279728583909395, 0.6224014365544661, 0.6254475218557116]



fig1 = plt.figure()
#fig1.set_facecolor('lightslategray')
ax1 = fig1.add_subplot(111)


ax1.plot(x, y_b4, 'r-', 
	marker='o', 
	markersize=14, 
	markerfacecolor='red',
	linewidth=3, 
	label="Full Back-off",
	mec='black')


ax1.plot(x, y_b2, '--', 
	marker="D", 
	c='darkgreen',
	markersize=10, 
	markerfacecolor='orange', 
	mec='black',
	mew=1,
	label="Back-off Depth 2")

ax1.plot(x, y_b3, 'm--', 
	marker="s", 
	markersize=10, 
	markerfacecolor='brown',
	mec='black',
	mew=1, 
	label="Back-off Depth 3")

ax1.grid(linestyle='-')
plt.xlim([39, 96])
plt.ylabel("BLEU Score")
plt.xlabel("Percentage of total data used as training")
legend = ax1.legend(loc='upper left', shadow=True)
#plt.savefig("ordered.pdf")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.grid(linestyle='-')

ax2.plot(x, y_b4, 'r-', 
	marker='o', 
	markersize=10, 
	markerfacecolor='red',
	linewidth=3, 
	label="Full Back-off",
	mec='black')

ax2.plot(x, y_b0, 'g--', 
	marker="s", 
	markersize=10, 
	markerfacecolor='purple', 
	label="No Back-off")

ax2.plot(x, y_b1, 'k--', 
	marker="^", 
	markersize=10, 
	markerfacecolor='blue', 
	mec='black',
	label="Back-off Depth 1")



legend = ax2.legend(loc='lower right', shadow=True)
plt.xlim([39, 96])
plt.ylim([0.1, 0.65])
plt.ylabel("BLEU Score")
plt.xlabel("Percentage of total data used as training")
plt.savefig("fullVSnone.pdf")
#plt.show()
