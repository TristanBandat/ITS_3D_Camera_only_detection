import logging
import os
from google.cloud import storage


def download_to_local(folder, blobs):
    logging.info('File download Started…. Wait for the job to complete.')
    # Create this folder locally if not exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Iterating through for loop one by one using API call
    for blob in blobs:
        logging.info('Blobs: {}'.format(blob.name))
        destination_uri = '{}/{}'.format(folder, blob.name)
        blob.download_to_filename(destination_uri)
        logging.info('Exported {} to {}'.format(blob.name, destination_uri))


def main():
    # logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    bucket_name = 'waymo_open_dataset_v_1_4_0'
    # table_id = ''
    # storage_client = storage.Client.from_service_account_json('/google-cloud/keyfile/service_account.json')
    storage_client = storage.Client.create_anonymous_client()
    # The “folder” where the files you want to download are
    # folder = '/google-cloud/download/{}'.format(table_id)
    folder = 'training'
    bucket = storage_client.bucket(bucket_name)
    # list all objects that satisfy the filter
    blobs = list(bucket.list_blobs())
    # download_to_local(folder, blobs)
    pass


if __name__ == '__main__':
    main()
