import os
import pickle
# import matplotlib.pyplot as plt


def main():
    with open('./final_data/waymo-data.pkl', 'rb') as f:
        data = pickle.load(f)
    for i, element in enumerate(data):
        tensor = data[i]['boxes']
        res = tensor.clone()
        res[tensor != 0] = 1
        data[i]['boxes'] = res
    with open('new_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    pass


if __name__ == '__main__':
    main()
