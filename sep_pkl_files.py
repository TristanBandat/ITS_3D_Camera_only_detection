import pickle
import os
from os.path import join


def main():
    file_ending = '.pkl'
    final_data_name = 'data_part2.pkl'
    file_to_sep = 'waymo-data_part1.pkl'
    sep_pos = 1910
    data_list = list()
    print('Checking for file...')
    # list all files in cwd
    all_files = os.listdir()
    # check for nonempty directory
    if len(all_files) != 0:
        # go through all files
        for filename in all_files:
            # check for correct file ending
            if filename.find(file_to_sep) != -1:
                print(f'Found file: {filename}')
                # Extract the data from the file
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                data_list = data[sep_pos:]
                break
        if len(data_list) != 0:
            with open((join(os.curdir, final_data_name)), 'wb') as f:
                pickle.dump(data_list, f)
    print('done.')
    pass


if __name__ == '__main__':
    main()
