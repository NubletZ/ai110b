import matplotlib.pyplot as plt
import numpy as np
import HillClimbing_Solutions as solution

# x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
# y = np.array([2, 3, 4, 5, 6], dtype=np.float32)
x = np.array([0,1,2,3,4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype = np.float32)

def optimize():
    l1 = [x[0],y[0]]
    l2 = [x[len(x)-1],y[len(y)-1]]
    m, c, err = solution.hillClimbing(l1,l2,x,y)
    p = [c, m]
    print('=== FINAL SOLUTION ===')
    print('solution: m={:f} c={:f}'.format(p[1],p[0]))
    print('loss=', err)
    return p

p = optimize()

# Plot the graph
y_predicted = list(map(lambda t: p[0]+p[1]*t, x))
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()