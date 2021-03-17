import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import optimize as optim
from PIL import Image as img
import matplotlib as mpl
mpl.use('TkAgg')


class LearningAlgorithm:
    def __init__(self):
        self.num_classes = None

    '''
        Name       :
        Purpose    : 
        Parameters :
        Return     :
        Notes      :
        '''
    def top_learning_algorithm(self):
        # Load data
        (input_data, output_data) = self.load_matrix_data()

        # Randomize Data Set
        #(input_data, output_data) = randomize_matrix_row_data(input_data, output_data)

        # Find the optimum values for theta
        parameter_values = self.one_vs_all(input_data, output_data, self.num_classes, 0.1)

        # Test for One versus all prediction
        result = self.predict_one_vs_all(parameter_values, input_data)
        for row in range(np.size(result)):
            a = result[row]
            b = output_data[row]
            if (result[row] == output_data[row]):
                result[row] = 1.0
            else:
                result[row] = 0.0
        training_accuracy = (np.sum(result) / np.size(result)) * 100.0
        print('\nTraining Set Accuracy : %.3f\n' % training_accuracy)

        self.store_parameters(parameter_values)

        return parameter_values

    ''' 
    Name       : load_matrix_data
    Purpose    : To read in the input and output data from csv files.
    Parameters : None
    Return     : Two matricies X which denotes the input data for the problem, and 
                    Y which denotes the output data for the problem.
    '''
    def load_matrix_data(self):
        data = np.loadtxt(open('input_and_output_data.csv'), delimiter=",", dtype='float32')
        X = data[:, 0: -1]
        y = data[:, -1]
        self.num_classes = len(np.unique(y))
        return X, y

    '''
    Name       : sigmoid_hypothesis
    Purpose    : To calculate the hypothesis for each example
    Parameters : theta : parameters, X : the design matrix
    Return     : g a column vector denoting the hypothesis calculation for each 
                    example
    Notes      :
                Before the sigmoid function can be called we need to multiply 
                    X * theta
                theta has dimensions: (number of features x 1)
                X has dimensions    : (number of examples x number of features)
                g has dimensions    : (number of examples x 1)
    '''
    def sigmoid_hypothesis(self, theta, X):
        # Perform matrix multiplication to obtain the sigmoid input vector
        z = np.matmul(X, theta)

        # Perform sigmoid calculations
        g = 1.0 / (1.0 + np.exp(-z))

        return g

    '''
    Name       : lr_cost_function_regularized
    Purpose    : Computes the cost for regularized logistic regression.
    Parameters : theta which are the weights, X which is the input values, y which
                    is the output values, lambdaConst which is the regularization
                    constant.
    Return     : One floating point value denoting the cost.
    Notes      :
                 theta has dimensions : (number of features x 1) 
                 X has dimensions     : (number of training examples x number of 
                                            features)
                 y has dimensions     : (number of examples x 1) 
                 hyp has dimensions   : (number of training examples x 1)
    '''
    def lr_cost_function_regularized(self, theta, X, y, lambdaConst):
        # Temp Theta
        temp_theta = theta.copy()

        # Number of training examples
        m = len(y)

        # Hypothesis calculation
        hyp = self.sigmoid_hypothesis(temp_theta, X)

        # Regularized Cost Function Calculation
        term_0 = np.matmul(np.transpose(-1.0 * y), np.log(hyp))
        term_1 = np.matmul(np.transpose(1.0 - y), np.log(1.0 - hyp))
        temp_theta[0] = 0
        term_2 = (lambdaConst / (2.0 * m)) * np.matmul(np.transpose(temp_theta),
                                                       temp_theta)
        J = (term_0 - term_1) * (1.0 / m) + term_2

        return J

    '''
    Name       : lr_gradient_regularized
    Purpose    : Computes the gradient for a logistic regression cost function
    Parameters : theta which are the weights, X which is the input values, y which 
                    is the output values, lambdaConst which 
                 is the regularization constant.
    Return     : grad which is a vector denoting the gradient values
    Notes      :
                theta has dimensions : (number of features x 1)
                X has dimensions     : (number of training examples x number of 
                                            features) 
                y has dimensions     : (number of examples x 1)
                hyp has dimensions   : (number of training examples x 1)
                grad has dimensions  : (number of features x 1) 
    
    '''
    def lr_gradient_regularized(self, theta, X, y, reg_const):
        # Temp theta
        temp_theta = theta.copy()

        # Number of training examples
        m = len(y)

        # Hypothesis calculation
        hyp = self.sigmoid_hypothesis(temp_theta, X)

        # Gradient Descent
        temp_theta[0] = 0
        X_transpose = np.transpose(X)
        inner = np.subtract(hyp, y)
        grad = (1.0 / m) * np.matmul(X_transpose, inner)
        reg_term = (reg_const / (1.0 * m)) * temp_theta
        grad = grad + reg_term

        return grad

    '''
    Name       : one_vs_all
    Purpose    : trains multiple logistic regression classifiers and returns all
                 the classifiers in a matrix all_theta, where the i-th row of all_theta 
                 corresponds to the classifier for label i.
    Parameters : X which is the input values, y which is the output values, num_labels
                 which is the number of classes we have, and lambdaConst which is 
                 the regularization constant.
    Return     :
    Notes      :
                 X has dimensions         : (number of training examples x number 
                                                of features)
                 y has dimensions         : (number of examples x 1)
                 all_theta had dimensions : (number of classes x  number of 
                                                features)
    '''
    def one_vs_all(self, X, y, num_labels, reg_const):
        m = np.size(X, 0)
        n = np.size(X, 1)

        # Initialize output matrix
        all_theta = np.zeros((num_labels, n + 1))

        # Add the bias unit to each input example as the first column
        ones_column_vector = np.zeros((m, 1)) + 1.0
        x_with_bias = np.concatenate((ones_column_vector, X), 1)

        # Initial Theta values
        initial_theta = np.zeros((n + 1, 1))

        # Generate classifiers for each class
        for num in range(num_labels):
            new_y = self.logical_array(y, num)
            res = optim.minimize(
                fun=self.lr_cost_function_regularized,
                x0=initial_theta,
                args=(x_with_bias, new_y, reg_const),
                method='CG',
                jac=self.lr_gradient_regularized,
                options={'gtol': 1e-9, 'maxiter': 100}
            )
            print(res.message)
            curr_theta = res.x
            all_theta[num][:] = curr_theta

        return all_theta

    '''
    Name       : logical_array
    Purpose    : To generate a logical array of 1 or 0 values based on if the 
                    current element in the given array is equal to the current 
                    number.
    Parameters : output_arr which is an array denoting the respective class the ith
                    element of the given array belongs to, curr_num which is the 
                    current class number.
    Return     : A logical array of one and zero values.
    Notes      : This function allows for one-versus-all multi-class classification
                    problems.
    '''
    def logical_array(self, output_arr, curr_num):
        logical_array = output_arr.copy()
        for row in range(np.size(logical_array)):
            if (logical_array[row] == curr_num):
                logical_array[row] = 1
            else:
                logical_array[row] = 0
        return logical_array

    '''
    Name       : predict_one_vs_all
    Purpose    : 
    Parameters : all_theta, which is a matrix denoting the learned parameters, X
                    which is a matrix denoting the input training data.
    Return     : predict which is a row vector denoting the class each example is
                    is most likely to belong to.
    Notes      :
    '''
    def predict_one_vs_all(self, all_theta, X):
        # Number of training examples
        m = np.size(X, 0)

        # Number of classes
        num_labels = np.size(all_theta, 0)

        # Initialize return array
        predict = np.zeros((m, 1))

        # Add ones to the X data matrix
        ones_column_vector = np.zeros((m, 1)) + 1.0
        x_with_bias = np.concatenate((ones_column_vector, X), 1)

        # Hypothesis calculation
        hyp = self.sigmoid_hypothesis(np.transpose(all_theta), x_with_bias)

        # Mapping the results to their respective class
        for row in range(np.size(hyp, 0)):
            max_index = 0
            max_val = -1
            for col in range(np.size(hyp, 1)):
                if (max_val < hyp[row][col]):
                    max_val = hyp[row][col]
                    max_index = col
            predict[row] = max_index

        return predict

    '''
    Name       : randomize_matrix_row_data
    Purpose    : To synchronously randomize the input and output data
    Parameters : input_data_matrix which is the input data matrix, output_data_vector which is the output data vector.
    Return     : random_input_data which denotes the randomized input data matrix, random_output_data which denotes the 
                 randomized output data matrix. 
    Notes      :
                random_input_data has dimensions:  (number of examples x number of features)
                random_output_data has dimensions: (number of examples x 1)
                The ith row of random_input_data corresponds to the ith row of random_output_data
                data_matrix is a concatenation of the input and output data column-wise
                data_matrix has dimensions (number of examples x number of features + 1)
                The plus 1 denotes the output column vector being added to the input data matrix
    '''
    @staticmethod
    def randomize_matrix_row_data(input_data_matrix, output_data_vector):
        # Concatenate input and output data into one data matrix
        data_matrix = np.concatenate((input_data_matrix, output_data_vector), axis=1)

        # Copy matrix data to a new matrix
        copy_data_matrix = np.array(data_matrix, copy=True)

        # Randomize data matrix
        np.random.shuffle(copy_data_matrix)

        # Matrix data
        (m, n) = copy_data_matrix.shape

        # Split data accordingly
        random_input_data = copy_data_matrix[:, 0:n-1]
        random_output_data = copy_data_matrix[:, n-1]

        return random_input_data, random_output_data

    '''
    Name       :
    Purpose    : 
    Parameters :
    Return     :
    Notes      :
    '''
    @staticmethod
    def store_parameters(theta_values):
        file_name = 'parameter_values.csv'

        # Open the current file and truncate the data
        param_file = open(file_name, 'w+', newline='')

        # Write parameter values
        with param_file:
            writer = csv.writer(param_file)
            writer.writerows(theta_values)

        # Close the parameter file
        param_file.close()

        return