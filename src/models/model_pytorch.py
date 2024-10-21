# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm.auto import trange, tqdm
#
# from src.models.deeplstmbimodel_pytorch import DeepLSTMBiModel
#
#
# class Model:
#     def __init__(
#             self,
#             input_size,
#             num_classes,
#             patch_size,
#             lstm_layer,
#             hidden_size,
#             number_block,
#             batch_size,
#             device="cuda",
#     ):
#
#         self.lstm = DeepLSTMBiModel(
#             input_size=input_size,
#             output_size=num_classes,
#             patch_size=patch_size,
#             lstm_layers=lstm_layer,
#             hidden_sizes=hidden_size,
#             number_block=number_block,
#         ).to(device)
#         self.device = device
#         self.batch_size = batch_size
#
#     def predict(self, data: torch.Tensor) -> torch.Tensor:
#         # Set the models to evaluation mode
#         self.lstm.eval()
#         with torch.no_grad():
#             data = data.to(self.device)
#             data_pred = self.lstm(data)
#             last_output = data_pred[:, -1, :]
#             acc = last_output.argmax(dim=1)
#         return acc
#
#     def test_model_lstm(self, test_loader):
#         # Set the models to evaluation mode
#         self.lstm.eval()
#         correct, total = 0, 0
#
#         with torch.no_grad():
#             for data, target in tqdm(test_loader, desc="Testing", leave=False):
#                 data, target = data.to(self.device), target.to(self.device)
#                 data_pred = self.lstm(data)
#                 last_output = data_pred[:, -1, :]
#                 correct += (last_output.argmax(dim=1) == target).sum().item()
#                 total += target.size(0)
#
#         test_accuracy = correct / total
#         return test_accuracy
#
#     def fit(
#             self,
#             X_train,
#             y_train,
#             X_val,
#             y_val,
#             learning_rate=0.001,
#             momentum=0.9,
#             epochs=100,
#             device="cuda",
#     ):
#         # Initialize the optimizer with the models parameters
#         optimizer = optim.Adam(
#             self.lstm.parameters(), lr=learning_rate, betas=(momentum, 0.999)
#         )
#         loss_function = nn.CrossEntropyLoss()
#
#         train_data = torch.utils.data.DataLoader(
#             torch.tensor(X_train, dtype=torch.float32),
#             torch.tensor(y_train, dtype=torch.long),
#         )
#
#         val_data = torch.utils.data.DataLoader(
#             torch.tensor(X_val, dtype=torch.float32),
#             torch.tensor(y_val, dtype=torch.long),
#         )
#
#         train_loader = torch.utils.data.DataLoader(
#             train_data, batch_size=self.batch_size, shuffle=True
#         )
#         val_loader = torch.utils.data.DataLoader(
#             val_data, batch_size=self.batch_size, shuffle=False
#         )
#
#         # Initialize training and validation loss and accuracy
#         training_loss_logger, validation_loss_logger = [], []
#         training_accuracy_logger, validation_accuracy_logger = [], []
#         print(f"Number of parameters: {sum(p.numel() for p in self.lstm.parameters())}")
#         # Train the models
#         train_acc = 0
#         val_acc = 0
#         # Initialize a progress bar to track epochs and display training and validation accuracies
#         pbar = trange(epochs, desc="Epoch", leave=False)
#         for epoch in pbar:
#             # Set the models to training mode
#             self.lstm.train()
#             correct, total = 0, 0
#
#             for data, target in tqdm(train_loader, desc="Training", leave=False):
#                 # Move the data and target to the device (GPU if available)
#                 data, target = data.to(device), target.to(device)
#
#                 # Forward pass through the models
#                 data_pred = self.lstm(data)
#                 # Calculate the loss
#                 loss = loss_function(data_pred, target)
#
#                 # Backpropagation
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#                 # Log the training loss and calculate the number of correct predictions
#                 training_loss_logger.append(loss.item())
#                 correct += (data_pred.argmax(dim=1) == target).sum().item()
#                 total += target.size(0)
#
#             # Calculate the training accuracy
#             train_acc = correct / total
#             training_accuracy_logger.append(train_acc)
#
#             # Validation loop
#             val_acc, val_loss = self._evaluate(val_loader, loss_function, device)
#             validation_loss_logger.append(val_loss)
#             validation_accuracy_logger.append(val_acc)
#
#             pbar.set_description(
#                 f"Epoch {epoch + 1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%"
#             )
#
#         return (
#             training_loss_logger,
#             validation_loss_logger,
#             training_accuracy_logger,
#             validation_accuracy_logger,
#         )
#
#     def _evaluate(self, data_loader, loss_function, device):
#         """Evaluate the model on a given data loader."""
#         self.lstm.eval()
#         correct, total, val_loss = 0, 0, 0.0
#
#         with torch.no_grad():
#             for data, target in tqdm(data_loader, desc="Validation", leave=False):
#                 data, target = data.to(device), target.to(device)
#                 data_pred = self.lstm(data)
#                 val_loss += loss_function(data_pred, target).item()
#                 correct += (data_pred.argmax(dim=1) == target).sum().item()
#                 total += target.size(0)
#
#         val_acc = correct / total
#         val_loss /= len(data_loader)
#         return val_acc, val_loss
#
#     def save_model(self, model_path):
#         # Save the models state dictionary
#         torch.save(self.lstm.state_dict(), model_path)
#
#     def load_model(self, model_path):
#         # Load the models state dictionary
#         self.lstm.load_state_dict(torch.load(model_path))
