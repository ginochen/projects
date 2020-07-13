
class NN(object):
    '''
    This codes are modified from https://github.com/SkalskiP/ILearnDeepLearning.py
    Main purpose of this is to build fully connected NN with numpy array 
    '''
    def __init__(self, dv=8):
        self.nn_architecture = [
            {"input_dim": dv, "output_dim": 4, "activation": "relu"},
            {"input_dim": 4, "output_dim": 6, "activation": "relu"},
            {"input_dim": 6, "output_dim": 6, "activation": "relu"},
            {"input_dim": 6, "output_dim": 4, "activation": "relu"},
            {"input_dim": 4, "output_dim": 1, "activation": "sigmoid"},
            ]       
        self.epochs = 10
        self.learning_rate = 0.001

    def init_layers(self, seed = 99):
        np.random.seed(seed)
        number_of_layers = len(self.nn_architecture)
        params_values = {}

        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]
        
            params_values['W' + str(layer_idx)] = np.random.randn(
                layer_output_size, layer_input_size) * 0.1
            params_values['b' + str(layer_idx)] = np.random.randn(
                layer_output_size, 1) * 0.1
        
        return params_values
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0,Z)

    def sigmoid_backward(self, dA, Z):
        return dA * self.sigmoid(Z) * (1 - sig) # dA = dL/dA

    def relu_backward(self, dA, Z):
        dZ = np.array(dA, copy = True)
        dZ[Z <= 0] = 0;
        return dZ;

    def single_layer_forward_propagation(self, A_prev, W_curr, b_curr, activation="relu"):
        Z_curr = np.dot(W_curr, A_prev) + b_curr
    
        if activation is "relu":
            activation_func = relu
        elif activation is "sigmoid":
            activation_func = sigmoid
        else:
            raise Exception('Non-supported activation function')
        
        return activation_func(Z_curr), Z_curr

    def full_forward_propagation(self, X, params_values):
        memory = {}
        A_curr = X
    
        for idx, layer in enumerate(self.nn_architecture):
            layer_idx = idx + 1
            A_prev = A_curr
        
            activ_function_curr = layer["activation"]
            W_curr = params_values["W" + str(layer_idx)]
            b_curr = params_values["b" + str(layer_idx)]
            A_curr, Z_curr = self.single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
            memory["A" + str(idx)] = A_prev
            memory["Z" + str(layer_idx)] = Z_curr
        
        return A_curr, memory

    def get_cost_value(self, Y_hat, Y):
        '''
         args:
          Y_hat[:output_nodes][:samples]
          Y[:output_nodes][:samples]

         Cross Entropy: 
         the probablity of the output q(y=1) = y_hat, q(y=0) = 1-y_hat 
         similarily p(y=1) = y and p(y=0) = 1-y
         
         Cross Entropy = H(p,q)= -sum_i p_i log q_i = -y log(y_hat) - (1-y)log(1-y_hat), where i is the outcome index

         Loss function:
         The average of the cross entropy over all samples.

         
         Why is cross-entropy related to MLE?
         To find the best parameter theta for deep learning model, we're essentially doing MLE for the parameters theta

         theta = argmax_theta 1/m \sum_i log(p_theta(x_i))  
               = argmax_theta E_p_data[log(p_theta(x_i))] 
               = - argmin_theta E_p_data[log(1/p_theta)] <---- that's the term to minimize,

         The difference to minimize between the data and model distribution 
         can be measured by KL divergence:
         
         D_KL(p_data||p_theta) = E_p_data[log(p_data) - log(p_theta)],
                                             ^ this term is a constant
         Since the left term is a constant, we just need to minimize the term 
         E_p_data[log(1/p_theta)] <----- minimize this term to shrink the difference between the two distributions!!!!!!!
                                         This term is the cross-entropy, so minimizing D_KL is essentially 
                                         minimizing cross-entropy and maximizing theta, i.e., MLE!!

        Summary:
         Minimizing cross-entropy term is the equivalent of finding the MLE for theta
        '''
        
        m = Y_hat.shape[1] # number of samples
        # averaged cross entropy over all samples 
        cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T)) # Y: truth, Y_hat: prediction, 
        return np.squeeze(cost)

    def get_accuracy_value(self, Y_hat, Y):
        Y_hat_ = convert_prob_into_class(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()

    def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
        '''
        
        Example:
        For 2 hidden layer binary classification, 
        the last sigmoid layer (layer [2])
        dA[2] = dL/dA[2] = 1/m * (-Y/A[2] + (1-Y)/(1-A[2]))
        dZ[2] = dL/dA[2] * dA[2]/dZ[2] 
              = (-Y/A[2] + (1-Y)/(1-A[2])) * A[2]'(Z[2]) 
              = (-Y/A[2] + (1-Y)/(1-A[2])) * (A[2]*(1-A[2])) 
              = (A[2]-Y)
        dW[2] = 1/m * dL/dA[2] * dA[2]/dZ[2] * dZ[2]/dW[2] # 1/m is from the averaged cost
              = 1/m * dZ[2] * d(W[2]*A[1]+b[2])/dW[2] 
              = 1/m * dZ[2] * A[1] 
        
        the layer before last layer (layer [1])
        dZ[1] = dL/dZ[2] * dZ[2]/dA[1] * dA[1]/dZ[1]
              = dZ[2] * d(W[2]*A[1]+b[2])/dA[1] * A[1]'(Z[1])
              = dZ[2] * W[2] * A[1]'(Z[1])
        dW[1] = 1/m * dL/dZ[1] * dZ[1]/dW[1]
              = 1/m * dZ[1] * d(W[1]*X+b[1])/dW[1]
              = 1/m * dZ[1] * X
              
        
        dW = 1/m dZ A
        '''
        m = A_prev.shape[1]
    
        if activation is "relu":
            backward_activation_func = relu_backward
        elif activation is "sigmoid":
            backward_activation_func = sigmoid_backward
        else:
            raise Exception('Non-supported activation function')
    
        dZ_curr = backward_activation_func(dA_curr, Z_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        return dA_prev, dW_curr, db_curr  

    def full_backward_propagation(Y_hat, Y, memory, params_values):
        grads_values = {}
        m = Y.shape[1]
        Y = Y.reshape(Y_hat.shape)
   
        dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
        for layer_idx_prev, layer in reversed(list(enumerate(self.nn_architecture))):
            layer_idx_curr = layer_idx_prev + 1
            activ_function_curr = layer["activation"]
        
            dA_curr = dA_prev
        
            A_prev = memory["A" + str(layer_idx_prev)] # 
            Z_curr = memory["Z" + str(layer_idx_curr)]
            W_curr = params_values["W" + str(layer_idx_curr)]
            b_curr = params_values["b" + str(layer_idx_curr)]
        
            dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
                dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)
        
            grads_values["dW" + str(layer_idx_curr)] = dW_curr
            grads_values["db" + str(layer_idx_curr)] = db_curr
    
        return grads_values  

    def update(params_values, grads_values):
        '''
        this is the simplest form of stochastic grdient descent w = w - a*dL/dw
        no minibatch, no Momentum or RMSprop
        '''
        for layer_idx, layer in enumerate(self.nn_architecture):
            params_values["W" + str(layer_idx)] -= self.learning_rate * grads_values["dW" + str(layer_idx)]        
            params_values["b" + str(layer_idx)] -= self.learning_rate * grads_values["db" + str(layer_idx)]

        return params_values

    def train(X, Y):
        params_values = init_layers(self.nn_architecture, 2)
        cost_history = []
        accuracy_history = []
    
        for i in range(self.epochs):
            Y_hat, cashe = full_forward_propagation(X, params_values, self.nn_architecture)
            cost = get_cost_value(Y_hat, Y)
            cost_history.append(cost)
            accuracy = get_accuracy_value(Y_hat, Y)
            accuracy_history.append(accuracy)
        
            grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, self.nn_architecture)
            params_values = update(params_values, grads_values, self.nn_architecture, self.learning_rate)
        
        return params_values, cost_history, accuracy_history
