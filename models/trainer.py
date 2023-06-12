""" Rohan Sanda 2023 """
from data import segmentsData, SegmentsDataBoth
from model import NeuralNetwork, cnn_1d, ResNet, cnn_1d_mod, NeuralNetworkMod
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import itertools
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#from torch.utils.tensorboard import SummaryWriter
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Trainer:
    """Define the Trainer Class, which handles training and validation of the model."""
    def __init__(self, model, train_params, data_fnames, model_type):
        # Define hyperparameters to tune
        self.input_size = 200
        if model_type == "ResNet" or model_type == "cnn_1d_mod" or model_type == "NeuralNetworkMod":
            #Multi-output SBP and DBP predictions
            self.output_size = 2
        else:
            #Single SBP OR DBP prediction
            self.output_size = 1
            
        self.model_type = model_type
        self.train_params = train_params
        self.tuned_model = model
        
        self.train_dataloader,  self.val_dataloader = self.get_dataloaders(data_fnames)

    def get_dataloaders(self, data_fnames):
        """I assume the data is stored in a pickle file. If not, change this function."""
        with open(data_fnames['train_arguments'], 'rb') as f:
            segments = pickle.load(f)
        
        with open(data_fnames['train_labels'], 'rb') as f:
            targets = pickle.load(f)
            
        """ If you want to use a different split, you'll have to change these hard-coded values.
            I just used a 70-20-10 split for all of my experiments."""
        
        #70-20-10 split: test set
        test_portion = 0.1
        num_rows = segments.shape[0]
        num_samples = test_portion*len(segments)
        random_indices = np.random.choice(int(num_rows), int(num_samples), replace=False)
        self.test_segments = segments[random_indices, :]
        self.test_sbp = targets[random_indices, 0]
        self.test_dbp = targets[random_indices, 1]
        
        segments = np.delete(segments, random_indices, axis=0)
        targets = np.delete(targets, random_indices, axis=0)
        targets_sbp = targets[:, 0]
        targets_dbp = targets[:, 1]
        
        #Format the data
        if self.model_type == "ResNet" or self.model_type == "cnn_1d_mod" or self.model_type == "NeuralNetworkMod":
            targets = np.stack((targets_sbp, targets_dbp), axis=1)
            dataset = SegmentsDataBoth(segments, targets_sbp, targets_dbp)
        else:
            dataset = segmentsData(segments, targets_sbp)
            
        #70-20-10 (train-val-test)
        train_portion = 0.8
        actual_train_portion = train_portion * (1-test_portion)
        train_size = int(actual_train_portion * len(dataset))
        val_size = len(dataset) - train_size
        train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.train_params["train_batch_size"], shuffle=True, num_workers=0)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=self.train_params["val_batch_size"], shuffle=False, num_workers=0)
        
        return train_dataloader, val_dataloader
    
    def train(self, verbose, iter_print):
        """This is for running a tuned model. See grid-search below to do hyperparameter tuning."""
        self.model = self.tuned_model
        self.optimizer = self.train_params["optimizers"][0](self.model.parameters(), lr=self.train_params['lr'][0])     
        self.criterion = nn.MSELoss()
        self.reg_lambda = self.train_params["reg_lambda"][0]
            
        # Things to store
        mse_losses = {'train_mse': [[]], 'val_mse': [[]]}
        mae_losses = {'train_mae': [[]], 'val_mae': [[]]}
        r2_scores = {'train_r2': [[]], 'val_r2': [[]]}
        losses = {'train_loss': [[]], 'val_loss': [[]]}
            
        for i in range(self.train_params["num_epochs"]):
            train_losses = []
            val_losses = []
            
            #Multi-output models
            if self.model_type == "ResNet" or self.model_type == "cnn_1d_mod" or self.model_type == "NeuralNetworkMod":
                train_loss, val_loss, train_mse_sbp, val_mse_sbp, train_mae_sbp, val_mae_sbp,train_r2_sbp,val_r2_sbp,train_mse_dbp,val_mse_dbp,train_mae_dbp,val_mae_dbp,train_r2_dbp,val_r2_dbp = self.train_epoch_resnet(verbose=verbose, iter_print=iter_print, iter=i)
                mse_losses["train_mse"].append([train_mse_sbp, train_mse_dbp])
                mse_losses["val_mse"].append([val_mse_sbp, val_mse_dbp])
                mae_losses["train_mae"].append([train_mae_sbp, train_mae_dbp])
                mae_losses["val_mae"].append([val_mae_sbp, val_mae_dbp])
                r2_scores["train_r2"].append([train_r2_sbp, train_r2_dbp])
                r2_scores["val_r2"].append([val_r2_sbp, val_r2_dbp])
                losses["train_loss"].append([train_loss])
                losses["val_loss"].append([val_loss])
            #Single output models   
            else:
                train_loss_i, val_loss_i, train_mae_i, val_mae_i, train_r2, val_r2 = self.train_epoch(verbose=verbose, iter_print=iter_print, iter=i)
                mse_losses["train_mse"].append([train_loss_i])
                mse_losses["val_mse"].append([val_loss_i])
                mae_losses["train_mae"].append([train_mae_i])
                mae_losses["val_mae"].append([val_mae_i])
                r2_scores["train_r2"].append([train_r2])
                r2_scores["val_r2"].append([val_r2])
        
        diastolic_r2, diastolic_mse, diastolic_mae, systolic_r2, systolic_mse, systolic_mae = self.predict()
        
        print("-----------TEST RESULTS----------")    
        print("Diastolic R2 Score:", diastolic_r2)
        print("Diastolic MSE:", diastolic_mse)
        print("Diastolic MAE:", diastolic_mae)
        print("Systolic R2 Score:", systolic_r2)
        print("Systolic MSE:", systolic_mse)
        print("Systolic MAE:", systolic_mae)
                
        
        with open("mse_losses.pickle", "wb") as f:
            pickle.dump(mse_losses, f)
        with open("mae_losses.pickle", "wb") as f:
            pickle.dump(mae_losses, f)
        with open("r2_scores.pickle", "wb") as f:
            pickle.dump(r2_scores, f)
        with open("losses.pickle", "wb") as f:
            pickle.dump(losses, f)
    
    def grid_search(self, verbose, iter_print):
        """Performs grid search on the hyperparameter combinations specified in config.py"""
        print("Grid search...")
        print_output = []
        # Generate all possible hyperparameter combinations
        if self.model_type == "NeuralNetwork":
            hyperparameter_combinations = itertools.product(self.train_params["optimizers"], self.train_params["lr"], self.train_params["hidden_sizes"], self.train_params["num_layers"], self.train_params["activation_functions"])
        elif self.model_type == "cnn_1d":
            hyperparameter_combinations = itertools.product(self.train_params["optimizers"], self.train_params["lr"])
        elif self.model_type == "cnn_1d_mod":
            hyperparameter_combinations = itertools.product(self.train_params["optimizers"], self.train_params["lr"], self.train_params["reg_lambda"])
        elif self.model_type == "NeuralNetworkMod":
            hyperparameter_combinations = itertools.product(self.train_params["optimizers"], self.train_params["lr"], self.train_params["reg_lambda"])
        else:
            hyperparameter_combinations = itertools.product(self.train_params["optimizers"], self.train_params["lr"], self.train_params["reg_lambda"])
            
        for hyperparameters in hyperparameter_combinations:
            if self.model_type == "NeuralNetwork":
                self.optimizer_class, self.learning_rate, self.hidden_size, self.num_layers, self.activation = hyperparameters
                self.model = NeuralNetwork(self.input_size, self.hidden_size, self.output_size, self.num_layers, self.activation)
            elif self.model_type == "cnn_1d":
                self.optimizer_class, self.learning_rate = hyperparameters
                self.model = cnn_1d().to(device)
            elif self.model_type == "cnn_1d_mod":
                self.optimizer_class, self.learning_rate, self.reg_lambda = hyperparameters
                self.model = cnn_1d_mod().to(device)
            elif self.model_type == "NeuralNetworkMod":
                self.optimizer_class, self.learning_rate, self.reg_lambda = hyperparameters
                self.model = NeuralNetworkMod().to(device)
            else:
                self.optimizer_class, self.learning_rate, self.reg_lambda = hyperparameters
                self.model = ResNet().to(device)
                        
            self.criterion = nn.MSELoss()
            self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.learning_rate)
            
            # Normal training loop below here
            mse_losses = {'train_mse': [[]], 'val_mse': [[]]}
            mae_losses = {'train_mae': [[]], 'val_mae': [[]]}
            r2_scores = {'train_r2': [[]], 'val_r2': [[]]}
            losses = {'train_loss': [[]], 'val_loss': [[]]}
                
            #writer = SummaryWriter(log_dir="./logs/")
            for i in range(self.train_params["num_epochs"]):
                train_losses = []
                val_losses = []
                
                if self.model_type == "ResNet" or self.model_type == "cnn_1d_mod" or self.model_type == "NeuralNetworkMod":
                    train_loss, val_loss, train_mse_sbp, val_mse_sbp, train_mae_sbp, val_mae_sbp,train_r2_sbp,val_r2_sbp,train_mse_dbp,val_mse_dbp,train_mae_dbp,val_mae_dbp,train_r2_dbp,val_r2_dbp = self.train_epoch_resnet(verbose=verbose, iter_print=iter_print, iter=i)
                    mse_losses["train_mse"].append([train_mse_sbp, train_mse_dbp])
                    mse_losses["val_mse"].append([val_mse_sbp, val_mse_dbp])
                    mae_losses["train_mae"].append([train_mae_sbp, train_mae_dbp])
                    mae_losses["val_mae"].append([val_mae_sbp, val_mae_dbp])
                    r2_scores["train_r2"].append([train_r2_sbp, train_r2_dbp])
                    r2_scores["val_r2"].append([val_r2_sbp, val_r2_dbp])
                    losses["train_loss"].append([train_loss])
                    losses["val_loss"].append([val_loss])
                    output = f"Epoch: {i}, Hyper parameters: {hyperparameters}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train SBP MSE: {train_mse_sbp}, Val SBP MSE: {val_mse_sbp}, Train SBP MAE: {train_mae_sbp}, Val SBP MAE: {val_mae_sbp}, Train SBP R2: {train_r2_sbp}, Val SBP R2: {val_r2_sbp}, Train DBP MSE: {train_mse_dbp}, Val DBP MSE: {val_mse_dbp}, Train DBP MAE: {train_mae_dbp}, Val DBP MAE: {val_mae_dbp}, Train DBP R2: {train_r2_dbp}, Val DBP R2: {val_r2_dbp}"
                    print_output.append(output)
                    print(output)
                else:
                    train_loss_i, val_loss_i, train_mae_i, val_mae_i, train_r2, val_r2 = self.train_epoch(verbose=verbose, iter_print=iter_print, iter=i)
                    mse_losses["train_mse"].append([train_loss_i])
                    mse_losses["val_mse"].append([val_loss_i])
                    mae_losses["train_mae"].append([train_mae_i])
                    mae_losses["val_mae"].append([val_mae_i])
                    r2_scores["train_r2"].append([train_r2])
                    r2_scores["val_r2"].append([val_r2])
                    output = f"Epoch: {i}, Hyper parameters: {hyperparameters}, Train Loss: {train_loss_i}, Val Loss: {val_loss_i}, Train MAE: {train_mae_i}, Val MAE: {val_mae_i}, Train R2: {train_r2}, Val R2: {val_r2}"
                    print_output.append(output)
                    
            with open("hyper_parameter_tuning.pickle", "wb") as f:
                pickle.dump(print_output, f)
            with open("losses.pickle", "wb") as f:
                pickle.dump(losses, f)

                
    def train_epoch_resnet(self, verbose=False, iter_print=1, iter=0):
        """Epoch trainer function for multi-output models"""
        train_loss = 0.0
        val_loss = 0.0
        train_predictions_sbp = []
        train_targets_sbp = []
        train_predictions_dbp = []
        train_targets_dbp = []
        val_predictions_sbp = []
        val_targets_sbp = []
        val_predictions_dbp = []
        val_targets_dbp = []

        # Train
        self.model.train()  # Set the model to train mode

        for batch in self.train_dataloader:
            inputs, targets_sbp, targets_dbp = batch
            #Unsqueeze if its not the NeuralNetwork/Mod
            inputs = inputs if self.model_type == 'NeuralNetworkMod' else inputs.unsqueeze(1)

            sbp, dbp = self.model(inputs.float())
            outputs_sbp = sbp # Systolic blood pressure prediction
            outputs_dbp = dbp  # Diastolic blood pressure prediction
            
            loss_sbp = self.criterion(outputs_sbp.squeeze(), targets_sbp.float())
            loss_dbp = self.criterion(outputs_dbp.squeeze(), targets_dbp.float())
            
            # Compute L1 regularization
            l1_reg = torch.tensor(0.0).to(device)
            for param in self.model.parameters():
                l1_reg += torch.sum(torch.abs(param))
            
            # Compute total loss with L1 regularization
            loss = loss_sbp + loss_dbp + self.reg_lambda * l1_reg
            
            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            
            train_predictions_sbp.extend(outputs_sbp.tolist())
            train_targets_sbp.extend(targets_sbp.tolist())
            train_predictions_dbp.extend(outputs_dbp.tolist())
            train_targets_dbp.extend(targets_dbp.tolist())

        # Validate
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, targets_sbp, targets_dbp = batch

                inputs = inputs if self.model_type == 'NeuralNetworkMod' else inputs.unsqueeze(1)
                
                sbp, dbp = self.model(inputs.float())
                outputs_sbp = sbp  # Systolic blood pressure prediction
                outputs_dbp = dbp  # Diastolic blood pressure prediction
                
                loss_sbp = self.criterion(outputs_sbp.squeeze(), targets_sbp.float())
                loss_dbp = self.criterion(outputs_dbp.squeeze(), targets_dbp.float())
                
                # Compute L1 regularization
                l1_reg = torch.tensor(0.0).to(device)
                for param in self.model.parameters():
                    l1_reg += torch.sum(torch.abs(param))
                
                # Compute total loss with L1 regularization
                loss = loss_sbp + loss_dbp + self.reg_lambda * l1_reg
                
                val_loss += loss.item()
                
                val_predictions_sbp.extend(outputs_sbp.tolist())
                val_targets_sbp.extend(targets_sbp.tolist())
                val_predictions_dbp.extend(outputs_dbp.tolist())
                val_targets_dbp.extend(targets_dbp.tolist())

        train_loss /= len(self.train_dataloader)
        val_loss /= len(self.val_dataloader)

        # Convert to NumPy arrays
        train_predictions_sbp = np.array(train_predictions_sbp)
        train_targets_sbp = np.array(train_targets_sbp)
        train_predictions_dbp = np.array(train_predictions_dbp)
        train_targets_dbp = np.array(train_targets_dbp)
        val_predictions_sbp = np.array(val_predictions_sbp)
        val_targets_sbp = np.array(val_targets_sbp)
        val_predictions_dbp = np.array(val_predictions_dbp)
        val_targets_dbp = np.array(val_targets_dbp)

        # Compute R2 score
        train_r2_sbp = r2_score(train_targets_sbp, train_predictions_sbp)
        val_r2_sbp = r2_score(val_targets_sbp, val_predictions_sbp)
        train_r2_dbp = r2_score(train_targets_dbp, train_predictions_dbp)
        val_r2_dbp = r2_score(val_targets_dbp, val_predictions_dbp)

        # Compute MAE
        train_mae_sbp = mean_absolute_error(train_targets_sbp, train_predictions_sbp)
        val_mae_sbp = mean_absolute_error(val_targets_sbp, val_predictions_sbp)
        train_mae_dbp = mean_absolute_error(train_targets_dbp, train_predictions_dbp)
        val_mae_dbp = mean_absolute_error(val_targets_dbp, val_predictions_dbp)
        
        # Compute MSE
        train_mse_sbp = mean_squared_error(train_targets_sbp, train_predictions_sbp)
        val_mse_sbp = mean_squared_error(val_targets_sbp, val_predictions_sbp)
        train_mse_dbp = mean_squared_error(train_targets_dbp, train_predictions_dbp)
        val_mse_dbp = mean_squared_error(val_targets_dbp, val_predictions_dbp)


        if verbose and (iter+1) % iter_print == 0:
            print("--------------------- Epoch ", iter, " ---------------------")
            print(f"Epoch: {iter}, Train Loss: {train_loss}, Val Loss: {val_loss}")
            print(f"Train SBP MSE: {train_mse_sbp}, Val SBP MSE: {val_mse_sbp}")
            print(f"Train SBP MAE: {train_mae_sbp}, Val SBP MAE: {val_mae_sbp}")
            print(f"Train SBP R2: {train_r2_sbp}, Val SBP R2: {val_r2_sbp}")
            print(f"Train DBP MSE: {train_mse_dbp}, Val DBP MSE: {val_mse_dbp}")
            print(f"Train DBP MAE: {train_mae_dbp}, Val DBP MAE: {val_mae_dbp}")
            print(f"Train DBP R2: {train_r2_dbp}, Val DBP R2: {val_r2_dbp}")

        return (
            train_loss,
            val_loss,
            train_mse_sbp,
            val_mse_sbp,
            train_mae_sbp,
            val_mae_sbp,
            train_r2_sbp,
            val_r2_sbp,
            train_mse_dbp,
            val_mse_dbp,
            train_mae_dbp,
            val_mae_dbp,
            train_r2_dbp,
            val_r2_dbp
        )


    def train_epoch(self, verbose=False, iter_print=1, iter=0):
        """The 1 epoch trainer function for single-output models.
           I stopped using this a while ago (switched only to multi-output models) so it may not
           be up to date. For instance, it doesn't have regularization """
        train_loss = 0.0
        val_loss = 0.0
        train_predictions = []
        train_targets = []
        val_predictions = []
        val_targets = []

        # Train
        self.model.train()  # Set the model to train mode
     
        for batch in self.train_dataloader:
            inputs, labels = batch
            inputs = inputs.unsqueeze(1) if self.model_type == 'cnn_1d' else inputs

            outputs = self.model(inputs.float())

            loss = self.criterion(outputs.squeeze(), labels.float())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            if len(outputs.size()) == 0:  # Handle single-value outputs
                train_predictions.append(outputs.item())
                train_targets.append(labels.item())
            else:
                train_predictions.extend(outputs.squeeze().tolist())
                train_targets.extend(labels.tolist())

        # Validate
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for batch in self.val_dataloader:
                inputs, labels = batch

                inputs = inputs.unsqueeze(1) if self.model_type == 'cnn_1d' else inputs
                
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs.squeeze(), labels.float())

                val_loss += loss.item()

                if len(outputs.size()) == 0:  # Handle single-value outputs
                    val_predictions.append(outputs.item())
                    val_targets.append(labels.item())
                else:
                    val_predictions.extend(outputs.squeeze().tolist())
                    val_targets.extend(labels.tolist())

        train_loss /= len(self.train_dataloader)
        val_loss /= len(self.val_dataloader)

        # Convert to NumPy arrays
        train_predictions = np.array(train_predictions)
        train_targets = np.array(train_targets)
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)

        # Compute R2 score
        train_r2 = r2_score(train_targets, train_predictions)
        val_r2 = r2_score(val_targets, val_predictions)

        # Compute MAE
        train_mae = mean_absolute_error(train_targets, train_predictions)
        val_mae = mean_absolute_error(val_targets, val_predictions)
        
        if verbose and (iter+1) % iter_print == 0:
            print("--------------------- Epoch ", iter, " ---------------------")
            print(f"Epoch: {iter}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train MAE: {train_mae}, Val MAE: {val_mae}, Train R2: {train_r2}, Val R2: {val_r2}")    
        
        return train_loss, val_loss, train_mae, val_mae, train_r2, val_r2

    def predict(self):
        # Load test data from pickle file
        test_data = self.test_segments

        # Extract diastolic and systolic blood pressure from labels
        systolic_bp = self.test_sbp
        diastolic_bp = self.test_dbp
        
        # Convert test data to torch.Tensor
        test_data_tensor = torch.from_numpy(test_data).float()

        # Move test data to GPU
        test_data_tensor = test_data_tensor.to('cuda')

        # Move the model to GPU
        self.model.to('cuda')

        # Perform predictions
        with torch.no_grad():
            if self.model_type == "cnn_1d_mod" or self.model_type == "ResNet":
                systolic_pred, diastolic_pred = self.model(test_data_tensor.unsqueeze(1))
                systolic_pred = systolic_pred.squeeze(1).cpu().numpy()
                diastolic_pred = diastolic_pred.squeeze(1).cpu().numpy()
            else:
                systolic_pred, diastolic_pred = self.model(test_data_tensor)
                # Convert predictions to numpy arrays
                systolic_pred = systolic_pred.cpu().numpy()
                diastolic_pred = diastolic_pred.cpu().numpy()

        # Calculate R2 score, MSE, and MAE for diastolic blood pressure
        diastolic_r2 = r2_score(diastolic_bp, diastolic_pred)
        diastolic_mse = mean_squared_error(diastolic_bp, diastolic_pred)
        diastolic_mae = mean_absolute_error(diastolic_bp, diastolic_pred)

        # Calculate R2 score, MSE, and MAE for systolic blood pressure
        systolic_r2 = r2_score(systolic_bp, systolic_pred)
        systolic_mse = mean_squared_error(systolic_bp, systolic_pred)
        systolic_mae = mean_absolute_error(systolic_bp, systolic_pred)
        
        # Save the numpy arrays
        np.save('train_segments.npy', self.test_segments)
        np.save('actual_sbp.npy', self.test_sbp)
        np.save('actual_dbp.npy', self.test_dbp)
        np.save('predicted_sbp.npy', systolic_pred)
        np.save('predicted_dbp.npy', diastolic_pred)


        return diastolic_r2, diastolic_mse, diastolic_mae, systolic_r2, systolic_mse, systolic_mae

