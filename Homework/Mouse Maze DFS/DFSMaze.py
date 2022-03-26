maze = [[1,1,1,1,1,1,1,1], 
        [1,1,0,1,0,1,0,1],
        [0,0,0,0,0,0,0,1],
        [1,0,1,1,1,1,0,1],
        [1,0,0,0,0,0,1,1],
        [1,1,1,1,1,0,1,1]]

start_point = [2,0]
end_point = [5,5]
visited = []
unvisited = []
mazeLenI = 6
mazeLenJ = 8

def printMap():
        for i in maze:
                for j in i:
                        if j == 1:
                                print('🟦', end='')
                        if j == 0:
                                print('⬜', end='')
                        if j == 2:
                                print('⚫', end='')
                print()
        print()

def findRoute(i, j):
        maze[i][j] = 2
        printMap()
        if i == 5 and j == 5:
                print(f'Finish point = {i}, {j}')
                return i, j
        else:
                if i-1 >= 0:
                        if maze[i-1][j] == 0:
                                unvisited.append([i-1,j])
                if i+1 < mazeLenI:
                        if maze[i+1][j] == 0:
                                unvisited.append([i+1,j])
                if j+1 < mazeLenJ:
                        if maze[i][j+1] == 0:
                                unvisited.append([i,j+1])
                if j-1 >= 0:
                        if maze[i][j-1] == 0:
                                unvisited.append([i,j-1])
                if len(unvisited) > 0:
                        nextPos = unvisited.pop()
                        print(f'unvisited = {unvisited}')
                        findRoute(nextPos[0], nextPos[1])

findRoute(start_point[0], start_point[1])