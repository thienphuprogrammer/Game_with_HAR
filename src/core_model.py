from src.data_pipeline.preprocessing.handle_dim_sequence import (
    pad_length,
    truncate_length,
)
from src.data_pipeline.preprocessing.handle_missing_value import (
    get_variance_of_metrics,
    handle_fill_nan_by_variance,
)
from src.models.bilstm_tensorflow import BiLSTMTensorflow


class CoreModel:
    def __init__(self, model_path, device="cuda"):
        # self.model = Model(device=device)
        # self.model.load_model(model_path=model_path)
        self.model = BiLSTMTensorflow(input_shape=[None, 35, 99], num_classes=6)
        self.model.load(model_path)

    def predict(self, clip_frame, max_dim):
        # Resize the size of clip
        X_temp = None
        if clip_frame.shape[0] < max_dim:
            X_temp = pad_length(clip_frame, max_dim)
        elif clip_frame.shape[0] > max_dim:
            X_temp = truncate_length(clip_frame, max_dim)
        else:
            X_temp = clip_frame

        X = X_temp

        # Fill nan by variance
        variance_metric = get_variance_of_metrics(X)
        for i in range(X.shape[1]):
            X[:, i] = handle_fill_nan_by_variance(X[:, i], variance_metric[i])
        # reshape the data into 4D tensor (1, X.shape[0], X.shape[1], X.shape[2])
        X = X.reshape(1, X.shape[0], X.shape[1], X.shape[2])
        # X = torch.from_numpy(X).float()
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])
        return self.model.predict(X)
