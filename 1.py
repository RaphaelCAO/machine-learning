import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-1,1,100)
y = x**2
plt.plot(x,y)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
plt.show()

print "second change on github"
