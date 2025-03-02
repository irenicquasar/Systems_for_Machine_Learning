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

# constructing topelitz matrix for weights 
topelitz_weights = [] 
for m in range(32):
    row = []
    for c in range(4):
        for h in range(5):
            for w in range(5):
                row.append(weights[m][c][h][w])
    topelitz_weights.append(row)    

# constructing toeplitz input matrix for input fmaps
topelitz_input_fmaps = []

for c in range(4):
    for h in range(5):
        for w in range(5):
            row = []
            for n in range(8):
                for x in range(14):
                    for y in range(14):
                        row.append(input_fmaps[n][c][2*x+h][2*y+w])
            topelitz_input_fmaps.append(row)

# multiplying topelitz weights and topelitz input fmaps to get topeletz output fmaps
topelitz_output_fmaps = []
for x in range(32):
    row = []
    for z in range(1568):
        sum_elements = 0
        for y in range(100):
            sum_elements += topelitz_weights[x][y] * topelitz_input_fmaps[y][z]
        row.append(sum_elements)
    topelitz_output_fmaps.append(row)

# convert output fmaps to topelitz output fmaps to check if both are same
output_fmaps_converted_to_topelitz = []
for m in range(32):
    row = []
    for n in range(8):
        for x in range(14):
            for y in range(14):
                row.append(output_fmaps[n][m][x][y])
    output_fmaps_converted_to_topelitz.append(row)

# check if both are same
import numpy as np
topelitz_output_fmaps = np.array(topelitz_output_fmaps)
topelitz_output_fmaps = np.round(topelitz_output_fmaps, 5)

output_fmaps_converted_to_topelitz = np.array(output_fmaps_converted_to_topelitz)
output_fmaps_converted_to_topelitz = np.round(output_fmaps_converted_to_topelitz, 5)

print(np.array_equal(topelitz_output_fmaps, output_fmaps_converted_to_topelitz))