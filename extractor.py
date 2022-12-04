import os
import time
import random
from simple_waymo_open_dataset_reader import WaymoDataFileReader


NUM_FRAMES_PER_RECORD = 3


def extract_frames(filename, num_frames):
    """
    Function to extract a specified number of frames randomly
    from given file.
    :param filename: Name of the file where the frames are extracted from
    :param num_frames: Number of frames to be extracted
    :return: Extracted frames in a list
    """
    frames_list = list()
    # open the file
    datafile = WaymoDataFileReader(filename)
    # generate a table of the offset of all frame records in the file
    table = datafile.get_record_table()
    # get the random frame id's
    num_frames_table = len(table)
    rand_numbers = random.sample(range(0, num_frames_table), num_frames)
    for i in rand_numbers:
        # find frame id in file
        datafile.seek(table[i])
        # get the frame
        frame = datafile.read_record()
        # save frame in the list
        frames_list.append(frame)

    return frames_list


def main():
    random.seed = 1234
    file_ending = 'tfrecord'
    while True:
        print('Checking for files...\t\tType CTRL+C to escape')
        # list all files in cwd
        all_files = os.listdir()
        # check for nonempty directory
        if len(all_files) != 0:
            # go through all files
            for filename in all_files:
                # check for correct file ending
                if filename.find(file_ending) != -1:
                    print(f'Found file: {filename}')
                    # Extract the frames from the file
                    frames = extract_frames(filename, NUM_FRAMES_PER_RECORD)
                    # TODO: save the frames somewhere

                    # TODO: uncomment statement when working
                    # os.remove(filename)
                    print('done.')

        time.sleep(1)
    pass


if __name__ == '__main__':
    main()
