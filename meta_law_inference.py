import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Define the number of layers in our simulated environment
NUM_LAYERS = 3

# Define the number of data points per layer
DATA_POINTS_PER_LAYER = 100

# Define the dimensionality of the energy signature
ENERGY_SIGNATURE_DIM = 5

# Define the dimensionality of the observer interaction
OBSERVER_INTERACTION_DIM = 3


def generate_simulated_data(layer_num, num_points):
    """
    Generates simulated data for a given layer.

    The data consists of:
    - Energy signatures (random vectors)
    - Observer interactions (random vectors)
    - Emergent law parameters (random values that depend on layer and interactions)

    Args:
        layer_num: The layer number (integer).
        num_points: The number of data points to generate.

    Returns:
        A tuple containing:
        - energy_signatures: A numpy array of shape (num_points, ENERGY_SIGNATURE_DIM)
        - observer_interactions: A numpy array of shape (num_points, OBSERVER_INTERACTION_DIM)
        - emergent_law_parameters: A numpy array of shape (num_points,)
    """

    energy_signatures = np.random.rand(num_points, ENERGY_SIGNATURE_DIM)
    observer_interactions = np.random.rand(num_points, OBSERVER_INTERACTION_DIM)

    # Simulate emergent law parameters based on layer and observer interaction
    emergent_law_parameters = np.zeros(num_points)
    for i in range(num_points):
        # Layer influence (makes higher layers have different parameter scales)
        layer_influence = layer_num * 0.5

        # Observer interaction influence (simulates how observation changes the law)
        interaction_influence = np.sum(observer_interactions[i]) * 0.2

        # Base value with some randomness
        base_value = random.uniform(0, 1)

        emergent_law_parameters[i] = base_value + layer_influence + interaction_influence

    return energy_signatures, observer_interactions, emergent_law_parameters


def create_and_train_model(energy_signatures, observer_interactions, emergent_law_parameters):
    """
    Creates and trains a linear regression model to predict emergent law parameters.

    Args:
        energy_signatures: The energy signatures (numpy array).
        observer_interactions: The observer interactions (numpy array).
        emergent_law_parameters: The emergent law parameters (numpy array).

    Returns:
        A trained LinearRegression model.
    """

    # Combine energy signatures and observer interactions into the input features
    X = np.concatenate((energy_signatures, observer_interactions), axis=1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, emergent_law_parameters, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model


def main():
    """
    Main function to test the hypothesis.
    """

    models = []
    layer_data = []  # Store data for cross-layer validation

    print("Generating and Training Models for Each Layer:")
    for layer in range(NUM_LAYERS):
        print(f"\nLayer {layer}:")
        energy_signatures, observer_interactions, emergent_law_parameters = generate_simulated_data(layer, DATA_POINTS_PER_LAYER)
        model = create_and_train_model(energy_signatures, observer_interactions, emergent_law_parameters)
        models.append(model)
        layer_data.append((energy_signatures, observer_interactions, emergent_law_parameters))


    print("\nCross-Layer Validation:")
    for layer_train in range(NUM_LAYERS):
        for layer_test in range(NUM_LAYERS):
            if layer_train != layer_test:
                print(f"\nTraining on Layer {layer_train}, Testing on Layer {layer_test}:")

                # Get data for training and testing
                energy_signatures_train, observer_interactions_train, emergent_law_parameters_train = layer_data[layer_train]
                energy_signatures_test, observer_interactions_test, emergent_law_parameters_test = layer_data[layer_test]

                # Combine features for training and testing
                X_train = np.concatenate((energy_signatures_train, observer_interactions_train), axis=1)
                X_test = np.concatenate((energy_signatures_test, observer_interactions_test), axis=1)

                # Train a new model on the training data
                model = LinearRegression()
                model.fit(X_train, emergent_law_parameters_train)

                # Predict on the test data
                y_pred = model.predict(X_test)

                # Evaluate the model
                mse = mean_squared_error(emergent_law_parameters_test, y_pred)
                print(f"Mean Squared Error: {mse}")


if __name__ == "__main__":
    main()