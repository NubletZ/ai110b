import numpy as np

def find_ab(l1, l2):
    """Line encoded as l=(x,y)."""
    m = float((l2[1] - l1[1])) / float(l2[0] - l1[0])
    c = (l2[1] - (m * l2[0]))
    return m, c

def neighbour(l1,l2,dataX,dataY,b):
    m, c = find_ab(l1, l2)
    err_Score = 0
    if b < 0:
        state = "down"
    else:
        state = "up"

    for i in range(0, len(dataX)):
        y_line = dataX[i] * m + c
        err_Score = err_Score + (y_line - dataY[i])**2
    print("--nb {:s} m:{:.3f} c:{:.3f} err_Score:{:.3f}".format(state, m, c, err_Score))
    print("l1:{:.3f}, {:.3f} | l2[1]:{:.3f}, {:.3f}".format(l1[0],l1[1],l2[0],l2[1]))
    print("m:{:.3f}, c:{:.3f}".format(m,c))
    print("err_Score:{:.5f}\n".format(err_Score))
    return m, c, err_Score

def hillClimbing(l1,l2,dataX,dataY,state=0):
    print("--current")
    m, c = find_ab(l1, l2)
    err_Score = 0
    for i in range(0, len(dataX)):
        y_line = dataX[i] * m + c
        print("y_line:{:.3f}, {:.3f} * {:.3f} + {:.3f}".format(y_line,dataX[i],m,c))
        err_Score = err_Score + (y_line - dataY[i])**2
    print("l1:{:.3f}, {:.3f} | l2[1]:{:.3f}, {:.3f}".format(l1[0],l1[1],l2[0],l2[1]))
    print("m:{:.3f}, c:{:.3f}".format(m,c))
    print("err_Score:{:.5f}\n".format(err_Score))

    if state != -1:
        mUp, cUp, errUp = neighbour(l1,(l2[0],l2[1]+0.01),dataX,dataY,1)
    if state != 1:
        mDown, cDown, errDown = neighbour(l1,(l2[0],l2[1]-0.01),dataX,dataY,-1)

    if err_Score == 0:
        return m, c, err_Score
    #Compare with its neighbour
    elif state != -1 and errUp <= err_Score and errUp < errDown:
        godown = False
        print("go up")
        return hillClimbing(l1,(l2[0],l2[1]+0.01),dataX,dataY,1)
    elif state != 1 and errDown <= err_Score:
        print("go down")
        goup = False
        return hillClimbing(l1,(l2[0],l2[1]-0.01),dataX,dataY,-1)
    else:
        return m, c, err_Score

'''
For testing only :)       
#dx = np.array([0, 1, 2, 3, 4], dtype=np.float32)
#dy = np.array([2, 3, 6, 5, 4], dtype=np.float32)
dx = np.array([0,1,2,3,4], dtype=np.float32)
dy = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype = np.float32)

x = np.array([dx[0], dy[0]], dtype=np.float32)
y = np.array([dx[1], dy[1]], dtype=np.float32)
m,c,err = hillClimbing(x,y,dx,dy)
print("Final solution: a={:.3f} b={:.3f}".format(m,c))
'''