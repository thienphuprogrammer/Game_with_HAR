from src.data_pipeline.preprocessing.handle_dim_sequence import *
from src.data_pipeline.preprocessing.handle_missing_value import *
from src.data_pipeline.preprocessing.video_pose_extractor import *

label_list = {"Jump": 0, "Kick": 1, "Punch": 2, "Left": 3, "Right": 4, "Stand": 5}


def load_har_dataset(
    dataset_path: str,
    max_dim,
    resize_dataset_video_path: str = None,
    dataset_csv_path: str = None,
    train_test="train",
    update=False,
):
    if dataset_csv_path is None:
        dataset_csv_path = os.path.join(resize_dataset_video_path, "UCF")
    resize_dataset_video_path = os.path.join(resize_dataset_video_path)

    if train_test == "train":
        dataset_path = os.path.join(dataset_path, "Train")
        resize_dataset_video_path = os.path.join(resize_dataset_video_path, "Train")
        dataset_csv_path = os.path.join(dataset_csv_path, "Train")
        if os.path.exists(os.path.join(dataset_csv_path, "X_train.npy")) and not update:
            print("Loading data from saved numpy arrays")
            X_train = np.load(os.path.join(dataset_csv_path, "X_train.npy"))
            y_train = np.load(os.path.join(dataset_csv_path, "y_train.npy"))
            return X_train, y_train
    else:
        dataset_path = os.path.join(dataset_path, "Test")
        resize_dataset_video_path = os.path.join(resize_dataset_video_path, "Test")
        dataset_csv_path = os.path.join(dataset_csv_path, "Test")
        if os.path.exists(os.path.join(dataset_csv_path, "X_test.npy")) and not update:
            print("Loading data from saved numpy arrays")
            X_test = np.load(os.path.join(dataset_csv_path, "X_test.npy"))
            y_test = np.load(os.path.join(dataset_csv_path, "y_test.npy"))
            return X_test, y_test

    # Resize videos and save them
    # resize_videos_and_save(dataset_path, resize_dataset_video_path)
    # Process videos and save data to csv
    X, y = process_videos(resize_dataset_video_path, dataset_csv_path)

    X_train_temp, y_train_temp = [], []

    # Pad all sequences to the same length
    for i, element in enumerate(X):
        padded_element = pad_and_truncate(np.array(element), max_dim)
        X_train_temp = X_train_temp + padded_element
        y_train_temp = y_train_temp + [y[i]] * len(padded_element)

    # Convert to numpy
    X_train_temp = np.array(X_train_temp)

    # Create empty 4D array with 3 channels
    X_train = np.full(
        (X_train_temp.shape[0], X_train_temp.shape[1], X_train_temp.shape[2], 3),
        np.nan,
        dtype=np.float32,
    )
    # Fill the 4D array with the 3 channels
    for i, element in enumerate(X_train_temp):
        for j, row in enumerate(element):
            for k, value in enumerate(row):
                # Check if the value is np.array([nan, nan, nan])
                if np.isnan(value).all():
                    X_train_temp[i][j][k] = np.array(
                        [np.nan, np.nan, np.nan], dtype=np.float32
                    )
                else:
                    X_train_temp[i][j][k] = value

                X_train[i][j][k] = X_train_temp[i][j][k][:3]

    variance_metric = np.empty(
        (X_train.shape[0], X_train.shape[2], 3), dtype=np.float32
    )
    for i, element in enumerate(X_train):
        sub_variance_metric = get_variance_of_metrics(element)
        variance_metric[i] = sub_variance_metric

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[2]):
            X_train[i][:, j] = handle_fill_nan_by_variance(
                X_train[i][:, j], variance_metric[i][j]
            )

    # Convert the labels to integers
    y_train_temp = [label_list[label] for label in y_train_temp]

    X_train = X_train.astype(np.float32)
    y_train_temp = np.array(y_train_temp).astype(np.float32)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train_temp.shape}")

    # Drop the NaN values
    X_new_train = []
    y_new_train = []
    for i in range(X_train.shape[0]):
        if np.isnan(X_train[i]).sum() == 0:
            X_new_train.append(X_train[i])
            y_new_train.append(y_train_temp[i])

    X_new_train = np.array(X_new_train)
    y_new_train = np.array(y_new_train)

    print(
        f"X_new_train shape: {X_new_train.shape}, y_new_train shape: {y_new_train.shape}"
    )
    print("Data loaded successfully")
    print("-" * 50)

    # if train_test == "train":
    #     np.save(os.path.join(dataset_csv_path, "X_train.npy"), X_new_train)
    #     np.save(os.path.join(dataset_csv_path, "y_train.npy"), y_new_train)
    # else:
    #     np.save(os.path.join(dataset_csv_path, "X_test.npy"), X_new_train)
    #     np.save(os.path.join(dataset_csv_path, "y_test.npy"), y_new_train)
    return X_new_train, y_new_train
