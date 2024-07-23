
import numpy as np

X = [2.1245833148065847, 2.124722203700274, 2.1045826176820386, 2.0951967875806066]
U =  [0.4998333703621417, 0.49988890123319635, 0.5154503195621867, 0.5226860940777577]

std_list = []
for i in range(len(X)):
    std_list += [np.std([X[i], U[i]])]
    
print(std_list)
print(max(std_list))