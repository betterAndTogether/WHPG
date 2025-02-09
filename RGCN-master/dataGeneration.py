import random

import numpy as np
def dataGeneration():
    data = []
    for i in range(900):
        head = random.randint(0, 127)
        relation = random.randint(0, 7)
        tail = random.randint(0, 127)
        data.append([head, relation, tail])
    return np.array(data)
