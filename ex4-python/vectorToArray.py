def vectorToArray (vector, hidden_layer_size, input_layer_size, num_labels):
    theta1 = vector[0:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size, (input_layer_size + 1))
    theta2 = vector[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, (hidden_layer_size +1))
    #theta1 = np.reshape(vector[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1), order='F')
    #theta2 = np.reshape(vector[hidden_layer_size * (input_layer_size + 1):], (num_labels, hidden_layer_size + 1), order='F')
    return theta1, theta2
