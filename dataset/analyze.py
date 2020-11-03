import numpy as np
import json
from matplotlib import pyplot as plt

if __name__ == '__main__':
    with open('submit.json', 'r') as f:
        data = json.load(f)
    print(len(data))
    prods = []
    for i, k in enumerate(data):
        print(i, k, data[k]['prob'])
        prods.append(data[k]['prob'])

    plt.hist(prods, 50)
    plt.show()
