# naive conv using 7 loops 

# construction of weights and input fmaps
# input fmaps
# N = 8 , C = 4 , H - 32 , W = 32
# chossing random number between 0  and 1 and assigning - or + randomly 
import random
 
input_fmaps = []
for n in range(8):
    fmap = []
    for c in range(4):
        row = []
        for h in range(32):
            col = []
            for w in range(32):
                col.append(random.choice([-1,1]) * random.random())
            row.append(col)
        fmap.append(row)
    input_fmaps.append(fmap)

# weights 
# M = 32 , C = 4 , H = 5 , W = 5 , 
weights = []
for m in range(32):
    fmap = []
    for c in range(4):
        row = []
        for h in range(5):
            col = []
            for w in range(5):
                col.append(random.choice([-1,1]) * random.random())
            row.append(col)
        fmap.append(row)
    weights.append(fmap)

# 7-loop naive implementation with stride length = 2 , average pooling and zero padding 
# output will be 8 x 32 x 14 x 14 ( N x M x H x W)
output_fmaps = []
# N = 8 , C = 4 , H - 32 , W = 32 , M = 32 , H = 5 , W = 5
for n in range(8):
    fmap = []
    for m in range(32):
        row = []
        for x in range(14):
            col = []
            for y in range(14):
                mac = 0
                #mac += biases[m]
                for i in range(5):
                    for j in range(5):
                        for c in range(4):
                            mac += input_fmaps[n][c][2*x+i][2*y+j] * weights[m][c][i][j]    
                col.append(mac)
            row.append(col)
        fmap.append(row)
    output_fmaps.append(fmap)

#print(len(output_fmaps), len(output_fmaps[0]), len(output_fmaps[0][0]), len(output_fmaps[0][0][0]))