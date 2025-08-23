import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Suppress TensorFlow Info messages (optional)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Data Loading and Preprocessing

# Create dummy XRD data for demonstration
def generate_dummy_xrd_data(num_samples=1000, pattern_length=256):
    """Generates dummy XRD data for testing.  Each sample
       is a noisy sine wave with varying frequency and amplitude."""
    data = []
    for _ in range(num_samples):
        amplitude = np.random.uniform(0.5, 1.5)  # Varying amplitude
        frequency = np.random.uniform(0.1, 0.5)  # Varying frequency
        phase = np.random.uniform(0, 2 * np.pi)  # Varying phase
        noise = np.random.normal(0, 0.05, pattern_length)  # Add some noise

        x = np.linspace(0, 10, pattern_length)  # X-axis values
        pattern = amplitude * np.sin(2 * np.pi * frequency * x + phase) + noise
        data.append(pattern)
    return np.array(data)

xrd_data = generate_dummy_xrd_data()  # Generate the dummy data
#xrd_data = pd.read_csv("xrd_data.csv", header=None).values  # Load from CSV if available

# Normalize the data
xrd_data = (xrd_data - np.min(xrd_data)) / (np.max(xrd_data) - np.min(xrd_data))

# Split into training and testing sets
X_train, X_test = train_test_split(xrd_data, test_size=0.2, random_state=42)

# Reshape the data for the CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# 2. VAE Model Definition
def define_vae(input_shape, latent_dim):
    """Defines the VAE model with convolutional layers."""

    # Encoder
    input_layer = Input(shape=input_shape)
    x = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    shape_before_flattening = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)

    # Latent space
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    encoder = Model(input_layer, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape_before_flattening[1] * shape_before_flattening[2], activation='relu')(latent_inputs)
    x = Reshape((shape_before_flattening[1], shape_before_flattening[2]))(x)
    x = Conv1D(128, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)  # Sigmoid for normalized output

    decoder = Model(latent_inputs, decoded, name='decoder')
    decoder.summary()

    # VAE model
    outputs = decoder(encoder(input_layer)[2])
    vae = Model(input_layer, outputs, name='vae')

    # Loss function
    reconstruction_loss = tf.keras.losses.mean_squared_error(K.flatten(input_layer), K.flatten(outputs))
    reconstruction_loss *= input_shape[0]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return vae, encoder, decoder, z_mean, z_log_var

# Hyperparameters
input_shape = (X_train.shape[1], 1)
latent_dim = 2  # Reduced latent dimension for easier visualization
epochs = 50
batch_size = 32

# Build VAE model
vae, encoder, decoder, z_mean, z_log_var = define_vae(input_shape, latent_dim)

# Compile the VAE
vae.compile(optimizer=Adam(learning_rate=1e-3))
vae.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# 3. Training the VAE
history = vae.fit(X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[early_stopping])

# Plot training history
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('VAE Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# 4. Generating XRD Patterns and Visualizing Results

# Generate patterns from random latent vectors
num_generated = 5
random_latent_vectors = np.random.normal(0, 1, size=(num_generated, latent_dim))
generated_patterns = decoder.predict(random_latent_vectors)

# Visualize generated patterns
plt.figure(figsize=(12, 6))
for i in range(num_generated):
    plt.subplot(1, num_generated, i + 1)
    plt.plot(generated_patterns[i].reshape(-1))
    plt.title(f"Generated Pattern {i+1}")
    plt.xlabel("2-theta")
    plt.ylabel("Intensity")
plt.tight_layout()
plt.show()


# Optionally, encode and visualize latent space
# First encode all test data to the latent space
z_mean_test, z_log_var_test, z_test = encoder.predict(X_test)

# Visualize the latent space
plt.figure(figsize=(8, 6))
plt.scatter(z_mean_test[:, 0], z_mean_test[:, 1], c='blue', alpha=0.5)
plt.title('Latent Space Visualization (Test Data)')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.colorbar(label='Data Point Density') # this might not correlate well with the graph for dummy data
plt.show()


print("VAE training and generation complete.  See plots for results.")