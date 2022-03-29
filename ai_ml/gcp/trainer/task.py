import argparse
import numpy as np
import os
import tempfile

from google.cloud import storage
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

n_features = 1 # Two features: y (previous values) and whether the date is a holiday
n_input_steps = 30 # Lookback window
n_output_steps = 7 # How many steps to predict forward

epochs = 1000 # How many passes through the data (early-stopping will cause training to stop before this)
patience = 5 # Terminate training after the validation loss does not decrease after this many epochs

def download_blob(bucket_name, source_blob_name, destination_file_name):
    '''Downloads a blob from the bucket.'''
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob " + source_blob_name + " downloaded to " + destination_file_name + ".")

def extract_bucket_and_prefix_from_gcs_path(gcs_path: str):
    '''Given a complete GCS path, return the bucket name and prefix as a tuple.

    Example Usage:

        bucket, prefix = extract_bucket_and_prefix_from_gcs_path(
            "gs://example-bucket/path/to/folder"
        )

        # bucket = "example-bucket"
        # prefix = "path/to/folder"

    Args:
        gcs_path (str):
            Required. A full path to a Google Cloud Storage folder or resource.
            Can optionally include "gs://" prefix or end in a trailing slash "/".

    Returns:
        Tuple[str, Optional[str]]
            A (bucket, prefix) pair from provided GCS path. If a prefix is not
            present, a None will be returned in its place.
    '''
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]
    if gcs_path.endswith("/"):
        gcs_path = gcs_path[:-1]

    gcs_parts = gcs_path.split("/", 1)
    gcs_bucket = gcs_parts[0]
    gcs_blob_prefix = None if len(gcs_parts) == 1 else gcs_parts[1]

    return (gcs_bucket, gcs_blob_prefix)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-uri',
        default=None,
        help='URL where the training files are located')
    args = parser.parse_args()
    print(args)
    return args

def main():
    args = get_args()
    bucket_name, blob_prefix = extract_bucket_and_prefix_from_gcs_path(args.data_uri)
    
    # Get the training data and convert back to np arrays
    local_data_dir = os.path.join(os.getcwd(), tempfile.gettempdir())
    files = ['x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy']
 
    for file in files:
        download_blob(bucket_name, os.path.join(blob_prefix,file), os.path.join(local_data_dir,file))

    X_train = np.load(local_data_dir + '/x_train.npy')
    y_train = np.load(local_data_dir + '/y_train.npy')
    X_test = np.load(local_data_dir + '/x_test.npy')
    y_test = np.load(local_data_dir + '/y_test.npy')
        
    # Build and train the model
    model = Sequential([
        LSTM(64, input_shape=[n_input_steps, n_features], recurrent_activation=None),
        Dense(n_output_steps)])

    model.compile(optimizer='adam', loss='mae')

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    _ = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=epochs, callbacks=[early_stopping])
    
    # Export the model
    model.save(os.environ["AIP_MODEL_DIR"])
    
if __name__ == '__main__':
    main()
