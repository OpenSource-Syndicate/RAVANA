import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import numpy as np
import pandas as pd
import time
import psutil

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define SCAE architecture
class SCAE(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(SCAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()  # Assuming input is scaled 0-1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Define standard Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, input_dim),
            nn.Sigmoid()  # Assuming input is scaled 0-1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Function to train an autoencoder (SCAE or standard)
def train_autoencoder(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        for data, _ in dataloader:  # DataLoader returns (data, labels), but we don't need labels
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
    return model


# Function to evaluate an autoencoder (SCAE or standard)
def evaluate_autoencoder(model, dataloader):
    model.eval()
    reconstruction_errors = []
    with torch.no_grad():
        for data, _ in dataloader:
            outputs = model(data)
            loss = torch.mean((outputs - data)**2, dim=1) # MSE Loss
            reconstruction_errors.extend(loss.cpu().numpy())
    return np.array(reconstruction_errors)


# Function to load and preprocess NAB dataset
def load_nab_dataset(file_path):
    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    df = df.fillna(method='ffill') #handle missing data
    data = df['value'].values.reshape(-1, 1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = (data - data.min()) / (data.max() - data.min()) # Scale between 0-1
    labels = np.zeros(len(data))
    anomalies_file = file_path.replace(".csv", "_anomalies.json") #NAB anomaly label convention
    try: # handle absence of anomaly labels.
        anomalies = pd.read_json(anomalies_file, typ='series')
        anomaly_indices = [df.index.get_loc(pd.to_datetime(x)) for x in anomalies.index]
        labels[anomaly_indices] = 1
    except FileNotFoundError:
        print(f"Anomaly labels file not found {anomalies_file}. Assuming no anomalies and proceeding.")

    return data, labels

# Function to load and preprocess KDD Cup 99 dataset
def load_kdd_dataset(file_path, normal_label='normal.'):
    df = pd.read_csv(file_path, header=None)
    labels = (df[41] != normal_label).astype(int).values  # Assuming attack labels are not 'normal.'
    data = df.iloc[:, :41].values
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = (data - data.min()) / (data.max() - data.min())  # Scale between 0-1
    return data, labels


# Function to evaluate anomaly detection results
def evaluate_results(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (tn + fp)
    return f1, fpr

# Main function
def main():
    # Datasets
    nab_file = "artificialNoAnomaly/art_daily_no_noise.csv" #Example NAB file.  Adjust path if necessary
    kdd_file = "kddcup.data_10_percent.csv" # Example KDD file. Adjust path if necessary

    # Model parameters
    input_dim = 1  # For NAB
    hidden_dim1 = 8
    hidden_dim2 = 4
    learning_rate = 0.001
    epochs = 5

    # --- NAB Dataset ---
    print("--- NAB Dataset ---")
    try:
        nab_data, nab_labels = load_nab_dataset(nab_file)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(nab_data, nab_labels, test_size=0.2, random_state=42)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32) # Unused, but kept consistent.
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Create DataLoaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


        # --- SCAE ---
        print("Training SCAE...")
        scae = SCAE(input_dim, hidden_dim1, hidden_dim2)
        optimizer_scae = optim.Adam(scae.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        start_time = time.time()
        scae = train_autoencoder(scae, train_loader, optimizer_scae, criterion, epochs)
        training_time = time.time() - start_time

        start_time = time.time()
        reconstruction_errors_scae = evaluate_autoencoder(scae, test_loader)
        inference_time = time.time() - start_time
        throughput = len(X_test) / inference_time #samples per second

        # Anomaly detection using reconstruction error threshold
        threshold_scae = np.percentile(reconstruction_errors_scae, 95)  # Adjust percentile as needed
        y_pred_scae = (reconstruction_errors_scae > threshold_scae).astype(int)
        f1_scae, fpr_scae = evaluate_results(y_test, y_pred_scae)

        model_size_scae = sum(p.numel() for p in scae.parameters() if p.requires_grad) # Count trainable params

        print(f"SCAE - F1: {f1_scae:.4f}, FPR: {fpr_scae:.4f}, Throughput: {throughput:.2f} samples/s, Model Size: {model_size_scae} parameters, Training Time: {training_time:.2f} s, Inference Time: {inference_time:.2f} s, Memory Usage: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")


        # --- Autoencoder ---
        print("Training Autoencoder...")
        autoencoder = Autoencoder(input_dim, hidden_dim1, hidden_dim2)
        optimizer_ae = optim.Adam(autoencoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        start_time = time.time()
        autoencoder = train_autoencoder(autoencoder, train_loader, optimizer_ae, criterion, epochs)
        training_time = time.time() - start_time

        start_time = time.time()
        reconstruction_errors_ae = evaluate_autoencoder(autoencoder, test_loader)
        inference_time = time.time() - start_time
        throughput = len(X_test) / inference_time

        # Anomaly detection using reconstruction error threshold
        threshold_ae = np.percentile(reconstruction_errors_ae, 95)  # Adjust percentile as needed
        y_pred_ae = (reconstruction_errors_ae > threshold_ae).astype(int)
        f1_ae, fpr_ae = evaluate_results(y_test, y_pred_ae)
        model_size_ae = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)

        print(f"Autoencoder - F1: {f1_ae:.4f}, FPR: {fpr_ae:.4f}, Throughput: {throughput:.2f} samples/s, Model Size: {model_size_ae} parameters, Training Time: {training_time:.2f} s, Inference Time: {inference_time:.2f} s, Memory Usage: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

        # --- Isolation Forest ---
        print("Training Isolation Forest...")
        iso_forest = IsolationForest(random_state=42)
        start_time = time.time()
        iso_forest.fit(X_train)
        training_time = time.time() - start_time

        start_time = time.time()
        y_pred_iso = iso_forest.predict(X_test)
        inference_time = time.time() - start_time
        throughput = len(X_test) / inference_time
        y_pred_iso = (y_pred_iso == -1).astype(int)  # Convert -1/1 to 1/0
        f1_iso, fpr_iso = evaluate_results(y_test, y_pred_iso)
        model_size_iso = 0  # Cannot easily determine model size for sklearn models
        print(f"Isolation Forest - F1: {f1_iso:.4f}, FPR: {fpr_iso:.4f}, Throughput: {throughput:.2f} samples/s, Model Size: N/A, Training Time: {training_time:.2f} s, Inference Time: {inference_time:.2f} s, Memory Usage: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")



        # --- One-Class SVM ---
        print("Training One-Class SVM...")
        oc_svm = OneClassSVM()
        start_time = time.time()
        oc_svm.fit(X_train)
        training_time = time.time() - start_time

        start_time = time.time()
        y_pred_svm = oc_svm.predict(X_test)
        inference_time = time.time() - start_time
        throughput = len(X_test) / inference_time
        y_pred_svm = (y_pred_svm == -1).astype(int)  # Convert -1/1 to 1/0
        f1_svm, fpr_svm = evaluate_results(y_test, y_pred_svm)
        model_size_svm = 0  # Cannot easily determine model size for sklearn models
        print(f"One-Class SVM - F1: {f1_svm:.4f}, FPR: {fpr_svm:.4f}, Throughput: {throughput:.2f} samples/s, Model Size: N/A, Training Time: {training_time:.2f} s, Inference Time: {inference_time:.2f} s, Memory Usage: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")


    except FileNotFoundError:
        print(f"NAB Dataset file not found: {nab_file}. Skipping NAB tests.")
        print("Please download and configure the correct file path. A sample dataset can be found in the NAB repository on Github.")

    # --- KDD Cup 99 Dataset ---
    print("--- KDD Cup 99 Dataset ---")
    try:
        kdd_data, kdd_labels = load_kdd_dataset(kdd_file)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(kdd_data, kdd_labels, test_size=0.2, random_state=42)

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32) # Unused, but kept consistent.
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


        # Create DataLoaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        # --- SCAE ---
        input_dim = kdd_data.shape[1]
        scae = SCAE(input_dim, hidden_dim1, hidden_dim2)
        optimizer_scae = optim.Adam(scae.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        print("Training SCAE...")
        start_time = time.time()
        scae = train_autoencoder(scae, train_loader, optimizer_scae, criterion, epochs)
        training_time = time.time() - start_time

        start_time = time.time()
        reconstruction_errors_scae = evaluate_autoencoder(scae, test_loader)
        inference_time = time.time() - start_time
        throughput = len(X_test) / inference_time

        # Anomaly detection using reconstruction error threshold
        threshold_scae = np.percentile(reconstruction_errors_scae, 95)  # Adjust percentile as needed
        y_pred_scae = (reconstruction_errors_scae > threshold_scae).astype(int)
        f1_scae, fpr_scae = evaluate_results(y_test, y_pred_scae)
        model_size_scae = sum(p.numel() for p in scae.parameters() if p.requires_grad)

        print(f"SCAE - F1: {f1_scae:.4f}, FPR: {fpr_scae:.4f}, Throughput: {throughput:.2f} samples/s, Model Size: {model_size_scae} parameters, Training Time: {training_time:.2f} s, Inference Time: {inference_time:.2f} s, Memory Usage: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")


        # --- Autoencoder ---
        print("Training Autoencoder...")
        autoencoder = Autoencoder(input_dim, hidden_dim1, hidden_dim2)
        optimizer_ae = optim.Adam(autoencoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        start_time = time.time()
        autoencoder = train_autoencoder(autoencoder, train_loader, optimizer_ae, criterion, epochs)
        training_time = time.time() - start_time

        start_time = time.time()
        reconstruction_errors_ae = evaluate_autoencoder(autoencoder, test_loader)
        inference_time = time.time() - start_time
        throughput = len(X_test) / inference_time

        # Anomaly detection using reconstruction error threshold
        threshold_ae = np.percentile(reconstruction_errors_ae, 95)  # Adjust percentile as needed
        y_pred_ae = (reconstruction_errors_ae > threshold_ae).astype(int)
        f1_ae, fpr_ae = evaluate_results(y_test, y_pred_ae)
        model_size_ae = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)

        print(f"Autoencoder - F1: {f1_ae:.4f}, FPR: {fpr_ae:.4f}, Throughput: {throughput:.2f} samples/s, Model Size: {model_size_ae} parameters, Training Time: {training_time:.2f} s, Inference Time: {inference_time:.2f} s, Memory Usage: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

        # --- Isolation Forest ---
        print("Training Isolation Forest...")
        iso_forest = IsolationForest(random_state=42)
        start_time = time.time()
        iso_forest.fit(X_train)
        training_time = time.time() - start_time


        start_time = time.time()
        y_pred_iso = iso_forest.predict(X_test)
        inference_time = time.time() - start_time
        throughput = len(X_test) / inference_time
        y_pred_iso = (y_pred_iso == -1).astype(int)  # Convert -1/1 to 1/0
        f1_iso, fpr_iso = evaluate_results(y_test, y_pred_iso)
        model_size_iso = 0  # Cannot easily determine model size for sklearn models
        print(f"Isolation Forest - F1: {f1_iso:.4f}, FPR: {fpr_iso:.4f}, Throughput: {throughput:.2f} samples/s, Model Size: N/A, Training Time: {training_time:.2f} s, Inference Time: {inference_time:.2f} s, Memory Usage: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

        # --- One-Class SVM ---
        print("Training One-Class SVM...")
        oc_svm = OneClassSVM()
        start_time = time.time()
        oc_svm.fit(X_train)
        training_time = time.time() - start_time

        start_time = time.time()
        y_pred_svm = oc_svm.predict(X_test)
        inference_time = time.time() - start_time
        throughput = len(X_test) / inference_time
        y_pred_svm = (y_pred_svm == -1).astype(int)  # Convert -1/1 to 1/0
        f1_svm, fpr_svm = evaluate_results(y_test, y_pred_svm)
        model_size_svm = 0  # Cannot easily determine model size for sklearn models
        print(f"One-Class SVM - F1: {f1_svm:.4f}, FPR: {fpr_svm:.4f}, Throughput: {throughput:.2f} samples/s, Model Size: N/A, Training Time: {training_time:.2f} s, Inference Time: {inference_time:.2f} s, Memory Usage: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")


    except FileNotFoundError:
        print(f"KDD Cup 99 Dataset file not found: {kdd_file}. Skipping KDD tests.")
        print("Please download and configure the correct file path.")


if __name__ == "__main__":
    main()