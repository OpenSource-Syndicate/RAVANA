import numpy as np

# Define the search space
SEARCH_SPACE = {
    'layer_types': ['conv', 'relu', 'maxpool', 'linear', 'sigmoid'],
    'kernel_sizes': [3, 5, 7],
    'num_filters': [16, 32, 64],
    'pool_sizes': [2, 3],
    'input_size': [32, 64, 128],  # Example input sizes for linear layers
    'output_size': [10, 100, 1000]  # Example output sizes for linear layers
}

def create_architecture(num_layers=3):
    """
    Generates a random neural network architecture.
    """
    architecture = []
    for _ in range(num_layers):
        layer_type = np.random.choice(SEARCH_SPACE['layer_types'])
        layer = {'type': layer_type}

        if layer_type == 'conv':
            layer['kernel_size'] = np.random.choice(SEARCH_SPACE['kernel_sizes'])
            layer['num_filters'] = np.random.choice(SEARCH_SPACE['num_filters'])
        elif layer_type == 'maxpool':
            layer['pool_size'] = np.random.choice(SEARCH_SPACE['pool_sizes'])
        elif layer_type == 'linear':
            layer['input_size'] = np.random.choice(SEARCH_SPACE['input_size'])
            layer['output_size'] = np.random.choice(SEARCH_SPACE['output_size'])

        architecture.append(layer)
    return architecture

def one_hot_encode_architecture(architecture):
    """
    Encodes a neural network architecture using one-hot encoding.
    """
    encoded_vectors = []
    for layer in architecture:
        layer_type = layer['type']
        layer_vector = []

        # Encode layer type
        layer_type_vector = [1 if layer_type == t else 0 for t in SEARCH_SPACE['layer_types']]
        layer_vector.extend(layer_type_vector)

        # Encode layer-specific parameters
        if layer_type == 'conv':
            kernel_size_vector = [1 if layer['kernel_size'] == k else 0 for k in SEARCH_SPACE['kernel_sizes']]
            num_filters_vector = [1 if layer['num_filters'] == f else 0 for f in SEARCH_SPACE['num_filters']]
            layer_vector.extend(kernel_size_vector)
            layer_vector.extend(num_filters_vector)
        elif layer_type == 'maxpool':
            pool_size_vector = [1 if layer['pool_size'] == p else 0 for p in SEARCH_SPACE['pool_sizes']]
            layer_vector.extend(pool_size_vector)
        elif layer_type == 'linear':
             input_size_vector = [1 if layer['input_size'] == i else 0 for i in SEARCH_SPACE['input_size']]
             output_size_vector = [1 if layer['output_size'] == o else 0 for o in SEARCH_SPACE['output_size']]
             layer_vector.extend(input_size_vector)
             layer_vector.extend(output_size_vector)
        else: # relu or sigmoid.  Add dummy vectors to keep length consistent
            layer_vector.extend([0]*len(SEARCH_SPACE['kernel_sizes']))
            layer_vector.extend([0]*len(SEARCH_SPACE['num_filters']))
            layer_vector.extend([0]*len(SEARCH_SPACE['pool_sizes']))
            layer_vector.extend([0]*len(SEARCH_SPACE['input_size']))
            layer_vector.extend([0]*len(SEARCH_SPACE['output_size']))



        encoded_vectors.append(layer_vector)

    # Flatten into a single vector
    flattened_vector = []
    for layer_vector in encoded_vectors:
        flattened_vector.extend(layer_vector)


    return np.array(flattened_vector)

def decode_architecture(encoded_vector, num_layers, layer_size):
    """
    Decodes the one-hot encoded vector back into a neural network architecture.
    """
    architecture = []
    for i in range(num_layers):
        layer_vector = encoded_vector[i * layer_size: (i + 1) * layer_size]

        # Decode layer type
        layer_type_index = np.argmax(layer_vector[:len(SEARCH_SPACE['layer_types'])])
        layer_type = SEARCH_SPACE['layer_types'][layer_type_index]

        layer = {'type': layer_type}

        if layer_type == 'conv':
            kernel_size_index = np.argmax(layer_vector[len(SEARCH_SPACE['layer_types']):len(SEARCH_SPACE['layer_types']) + len(SEARCH_SPACE['kernel_sizes'])])
            num_filters_index = np.argmax(layer_vector[len(SEARCH_SPACE['layer_types']) + len(SEARCH_SPACE['kernel_sizes']):len(SEARCH_SPACE['layer_types']) + len(SEARCH_SPACE['kernel_sizes']) + len(SEARCH_SPACE['num_filters'])])
            layer['kernel_size'] = SEARCH_SPACE['kernel_sizes'][kernel_size_index]
            layer['num_filters'] = SEARCH_SPACE['num_filters'][num_filters_index]
        elif layer_type == 'maxpool':
            pool_size_index = np.argmax(layer_vector[len(SEARCH_SPACE['layer_types']):len(SEARCH_SPACE['layer_types']) + len(SEARCH_SPACE['pool_sizes'])])
            layer['pool_size'] = SEARCH_SPACE['pool_sizes'][pool_size_index]
        elif layer_type == 'linear':
            input_size_index = np.argmax(layer_vector[len(SEARCH_SPACE['layer_types']):len(SEARCH_SPACE['layer_types']) + len(SEARCH_SPACE['input_size'])])
            output_size_index = np.argmax(layer_vector[len(SEARCH_SPACE['layer_types']) + len(SEARCH_SPACE['input_size']):len(SEARCH_SPACE['layer_types']) + len(SEARCH_SPACE['input_size']) + len(SEARCH_SPACE['output_size'])])
            layer['input_size'] = SEARCH_SPACE['input_size'][input_size_index]
            layer['output_size'] = SEARCH_SPACE['output_size'][output_size_index]
        architecture.append(layer)

    return architecture


# Test the encoding and decoding process
if __name__ == '__main__':
    num_layers = 3
    architecture = create_architecture(num_layers)
    print("Original Architecture:")
    print(architecture)

    layer_size = len(SEARCH_SPACE['layer_types']) + len(SEARCH_SPACE['kernel_sizes']) + len(SEARCH_SPACE['num_filters']) + len(SEARCH_SPACE['pool_sizes']) + len(SEARCH_SPACE['input_size']) + len(SEARCH_SPACE['output_size'])
    encoded_vector = one_hot_encode_architecture(architecture)
    print("\nEncoded Vector:")
    print(encoded_vector)
    print("Length of encoded vector:", len(encoded_vector))

    decoded_architecture = decode_architecture(encoded_vector, num_layers, layer_size)
    print("\nDecoded Architecture:")
    print(decoded_architecture)

    # Verify that the decoded architecture is the same as the original
    print("\nVerification:")
    print("Original architecture == Decoded architecture: ", architecture == decoded_architecture)

    # Conclusion based on the test
    print("\nConclusion:")
    if architecture == decoded_architecture:
        print("The test shows that one-hot encoding can effectively represent and reconstruct neural network architectures within the defined search space.")
        print("Therefore, the hypothesis is supported by the test.")
    else:
        print("The test shows that one-hot encoding failed to reconstruct the neural network architecture exactly.")
        print("Therefore, the hypothesis is rejected by the test. Debugging is needed.")