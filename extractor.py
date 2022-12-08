import os
import time
import random
import pickle
from os.path import join
from waymo_open_dataset import dataset_pb2 as open_dataset
import tensorflow as tf
from compress_data import compress_frame_list

NUM_FRAMES_PER_RECORD = 20


def extract_frames(filename, num_frames):
    """
    Function to extract a specified number of frames randomly
    from given file.
    :param filename: Name of the file where the frames are extracted from
    :param num_frames: Number of frames to be extracted
    :return: Extracted frames in a list
    """
    frames_list = list()
    # open dataset
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    # iterate over shuffled dataset
    for i, data in enumerate(dataset.shuffle(200)):
        # parse data
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # append frame to list
        frames_list.append(frame)
        # check if enough frames are extracted
        if i == num_frames - 1:
            break

    return frames_list


def saveFramesAsPickle(frames : list, pklName):
    # check if there is an existing pickle file
    data = []
    path = join(os.curdir, pklName)

    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(path, "wb") as f:
            print('Pickle file created, since not existing yet')

    data.extend(frames)

    with open(path, "wb") as f:
        pickle.dump(data, f)


    #try:
    #    # load frames from pickle
    #    with open(path, "rb") as f:
    #        pkl = pickle.load(f)
    #    pkl.extend(frames)
    #    f = open(path, 'wb')
    #except FileNotFoundError:
    #    f = open(path, 'wb')
    #pickle.dump(pkl, f)
    #f.close()


def main():
    random.seed = 1234
    file_ending = 'tfrecord'
    wrong_file_ending = 'gstmp'
    while True:
        print('Checking for files...\t\tType CTRL+C to escape')
        # list all files in cwd
        all_files = os.listdir()
        # check for nonempty directory
        if len(all_files) != 0:
            # go through all files
            for filename in all_files:
                # check if the file is still downloading
                if filename.find(wrong_file_ending) != -1:
                    continue
                # check for correct file ending
                if filename.find(file_ending) != -1:
                    print(f'Found file: {filename}')
                    # Extract the frames from the file
                    frames = extract_frames(filename, NUM_FRAMES_PER_RECORD)
                    # compress the frames
                    comp_data = compress_frame_list(frames, compression_factor=6)
                    # Save the frames as a pickle
                    saveFramesAsPickle(comp_data, 'data_part9_10.pkl')
                    # Delete the file
                    # os.remove(filename)
                    print('done.')

        time.sleep(10)
    pass


if __name__ == '__main__':
    main()
