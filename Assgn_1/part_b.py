# Toeplitx conv

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

# Toeplitz conversion
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

#print(len(topelitz_output_fmaps), len(topelitz_output_fmaps[0]))