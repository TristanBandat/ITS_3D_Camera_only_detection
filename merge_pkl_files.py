import pickle
import os
from os.path import join


def main():
    file_ending = 'pkl'
    final_data_name = 'waymo-data.pkl'
    data_list = list()
    print('Checking for files...')
    # list all files in cwd
    all_files = os.listdir()
    # check for nonempty directory
    if len(all_files) != 0:
        # go through all files
        for filename in all_files:
            # check for correct file ending
            if filename.find(file_ending) != -1:
                print(f'Found file: {filename}')
                # Extract the data from the file
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                data_list.extend(data)
        if len(data_list) != 0:
            with open((join(os.curdir, final_data_name)), 'wb') as f:
                pickle.dump(data, f)
    print('done.')
    pass


if __name__ == '__main__':
    main()
