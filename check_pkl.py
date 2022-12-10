import pickle
import os


def main():
    with open('./final_data/waymo-data.pkl', 'rb') as f:
        data = pickle.load(f)
    # all_data = list()
    # for element in data:
    #     all_data.extend(element)
    # with open('./waymo-data_part4_comp2.pkl', 'wb') as f:
    #     pickle.dump(all_data, f)
    pass


if __name__ == '__main__':
    main()
