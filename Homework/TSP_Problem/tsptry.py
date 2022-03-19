import random

def loss(ans):
    total = 0
    for i in range(num_citys-1):
        total += distance[ans[i]-1][ans[i+1]-1]
    total += distance[ans[num_citys-1]-1][ans[0]-1]
    return total

file = "datasets/five_d.txt"
num_citys = 15
best = 291
distance = [
    [0,29,82,46,68,52,72,42,51,55,29,74,23,72,46],
    [29,0,55,46,42,43,43,23,23,31,41,51,11,52,21],
    [82,55,0,68,46,55,23,43,41,29,79,21,64,31,51],
    [46,46,68,0,82,15,72,31,62,42,21,51,51,43,64],
    [68,42,46,82,0,74,23,52,21,46,82,58,46,65,23],
    [52,43,55,15,74,0,61,23,55,31,33,37,51,29,59],
    [72,43,23,72,23,61,0,42,23,31,77,37,51,46,33],
    [42,23,43,31,52,23,42,0,33,15,37,33,33,31,37],
    [51,23,41,62,21,55,23,33,0,29,62,46,29,51,11],
    [55,31,29,42,46,31,31,15,29,0,51,21,41,23,37],
    [29,41,79,21,82,33,77,37,62,51,0,65,42,59,61],
    [74,51,21,51,58,37,37,33,46,21,65,0,61,11,55],
    [23,11,64,51,46,51,51,33,29,41,42,61,0,62,23],
    [72,52,31,43,65,29,46,31,51,23,59,11,62,0,59],
    [46,21,51,64,23,59,33,37,11,37,61,55,23,59,0]
]

num_steps = 1000
ans = [i for i in range(1, num_citys+1)]
random.shuffle((ans))
diff = 1
n=1
for step in range(num_steps):
    l = loss(ans)
    isforward = False
    for i in range(1, num_citys-diff-n): 
        tempans = ans[:] #代表陣列所有數值
        temploss = 0
        tempans[i:i+n], tempans[i+diff:i+diff+n] = tempans[i+diff:i+diff+n], tempans[i:i+n]
        temploss = loss(tempans)
        if temploss < l:
            isforward = True
            l = temploss
            ans = tempans[:]
            print(ans, l, diff, n)
            diff = 1
            n = 1
            break
    if not isforward:
        diff += 1
        if diff > num_citys-1:
            n += 1
    if l == best or n > num_citys-1:
        print("we did it")
        break
    

print(ans)
print(loss(ans))
# for i in range(num_citys):
#     for j in range(num_citys):
#         print(distance[i][j], end="\t")
#     print()