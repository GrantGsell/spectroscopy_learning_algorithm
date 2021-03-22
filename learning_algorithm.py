import numpy as np
import csv
from scipy import optimize as optim
import matplotlib.pyplot as plt

class LearningAlgorithm:
    def __init__(self):
        self.num_classes = None

    """
    Name       :
    Purpose    : 
    Parameters :
    Return     :
    Notes      :
    """
    def top_learning_algorithm(self):
        # Load data
        (input_data, output_data) = self.load_matrix_data()

        # Randomize Data Set
        (input_data, output_data) = self.randomize_matrix_row_data(input_data, output_data)

        # Split data set into 3 data sets
        training_set, cv_set, test_set = self.split_original_training_set(input_data, output_data)

        # Split each data set into input output
        training_set_input, training_set_output = self.split_data_set_into_i_o(training_set)
        cv_set_input, cv_set_output = self.split_data_set_into_i_o(cv_set)
        test_set_input, test_set_output = self.split_data_set_into_i_o(test_set)

        # Find the optimal regularization constant
        regularization_constant = self.lambda_selection(training_set_input, training_set_output, cv_set_input,
                                                  cv_set_output)

        # Find the optimum values for theta
        parameter_values = self.one_vs_all(input_data, output_data, self.num_classes, 0.1)

        # Test for One versus all prediction
        result = self.predict_one_vs_all(parameter_values, input_data)

        # Metrics for the given set of parameters
        self.metrics(result, output_data)

        # Store the optimized parameters in csv
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
        X, y = self.split_data_set_into_i_o(data)

        #X = data[:, 0: -1]
        #y = data[:, -1]

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

    """
    Name       :
    Purpose    : 
    Parameters :
    Return     :
    Notes      :
    """
    def train_logistic_regression(self, theta, X, y):
        result = self.predict_one_vs_all(theta, X)

        for row in range(np.size(result)):
            a = result[row]
            b = y[row]
            if result[row] == y[row]:
                result[row] = 1.0
            else:
                result[row] = 0.0
        accuracy = (np.sum(result) / np.size(result)) * 100.0
        return accuracy


    """
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
    """
    def randomize_matrix_row_data(self, input_data_matrix, output_data_vector):
        # Ensure the output vector is defined by a 2d tuple
        m = output_data_vector.shape
        output_data_vector = np.reshape(output_data_vector, (m[0], 1))

        # Concatenate input and output data into one data matrix
        data_matrix = np.concatenate((input_data_matrix, output_data_vector), axis=1)

        # Copy matrix data to a new matrix
        copy_data_matrix = np.array(data_matrix, copy=True)

        # Randomize data matrix
        np.random.shuffle(copy_data_matrix)

        # Matrix data
        (m, n) = copy_data_matrix.shape

        # Split data accordingly
        #random_input_data, random_output_data = self.split_data_set_into_i_o(copy_data_matrix)
        random_input_data = copy_data_matrix[:, 0:n-1]
        random_output_data = copy_data_matrix[:, n-1]

        return random_input_data, random_output_data

    """
    Name       :
    Purpose    : 
    Parameters :
    Return     :
    Notes      :
    """
    @staticmethod
    def split_data_set_into_i_o(data_set):
        data_set_input = data_set[:, 0:-1]
        data_set_output = data_set[:, -1]

        # Ensure data_set_output is defined by a 2d tuple
        #m = data_set_output.shape
        #data_set_output = np.reshape(data_set_output, (m[0], 1))

        return data_set_input, data_set_output

    """
    Name       :
    Purpose    : 
    Parameters :
    Return     :
    Notes      :
    """
    @staticmethod
    def split_original_training_set(data_set_input, data_set_output):
        # Combine input and output data
        m = data_set_output.shape
        data_set_output = np.reshape(data_set_output, (m[0], 1))
        original_training_set = np.concatenate((data_set_input, data_set_output), axis=1)

        # Number of training examples
        m = len(original_training_set)

        # Print the number of training examples
        print("There are %d training examples in the original training set.\n" % m)

        # Set indices for each of the three data sets
        training_set_lower_index = 0
        training_set_upper_index = int(m * 0.6)
        cv_set_lower_index = training_set_upper_index
        cv_set_upper_index = int(m * 0.2) + cv_set_lower_index
        test_set_lower_index = cv_set_upper_index
        test_set_upper_index = int(m * 0.2) + test_set_lower_index

        # Determine if any examples were missed
        delta = m - test_set_upper_index
        if delta > 0:
            training_set_upper_index += delta
            cv_set_lower_index += delta
            cv_set_upper_index += delta
            test_set_lower_index += delta
            test_set_upper_index += delta


        # Create the three data sets
        training_set = original_training_set[training_set_lower_index:training_set_upper_index][:]
        cv_set = original_training_set[cv_set_lower_index:cv_set_upper_index][:]
        test_set = original_training_set[test_set_lower_index:test_set_upper_index][:]

        # Obtain the number of training examples in each set
        m_training_set = np.shape(training_set)[0]
        m_cv_set = np.shape(cv_set)[0]
        m_test_set = np.shape(test_set)[0]
        total = m_training_set + m_cv_set + m_test_set

        # Create string format specifiers
        header_fs = "{head0:^25.25s} | {head1:^25.25s} | {head2:^25.25s} | {head3:^25.25s}"
        set_fs = "{col0:^25d} | {col1:^25d} | {col2:^25d} | {col3:^25d}"
        underline_fs = "{:-^100}"

        # Output set data
        print("Number of examples in each training set: \n")
        print(header_fs.format(head0="Training Set", head1="Cross Validation Set", head2="Test Set", head3="Total"))
        print(underline_fs.format(''))
        print(set_fs.format(col0=m_training_set, col1=m_cv_set, col2=m_test_set, col3=total))

        return training_set, cv_set, test_set

    """
    Name       :
    Purpose    : 
    Parameters :
    Return     :
    Notes      :
    """
    def lambda_selection(self, train_set_input, train_set_output, cv_set_input, cv_set_output):
        # Lambda Value Upper bound, step size, vector size
        lambda_upper = 0.5 #1.0 #10.0
        lambda_step = 0.05
        lambda_size = int(lambda_upper / lambda_step)

        # Lambda Vector values
        lambda_vector = np.zeros((lambda_size, 1))
        val = 0.0
        for j in range(lambda_size):
            lambda_vector[j] = val
            val += lambda_step

        # Length of lambda vector
        ll, temp = lambda_vector.shape

        # Return array initialization
        train_f_score = np.zeros((ll, self.num_classes))
        cross_validation_f_score = np.zeros((ll, self.num_classes))

        # Generate Training error values based on lambda values
        for i in range(ll):
            # Obtain the optimal values for theta using regularization
            optim_thetas = self.one_vs_all(train_set_input, train_set_output, self.num_classes, (lambda_vector[i])[0])

            # Make predictions for training set
            prediction_ts = self.predict_one_vs_all(optim_thetas, train_set_input)

            # Make predictions for CV set
            prediction_cv = self.predict_one_vs_all(optim_thetas, cv_set_input)

            # Run metrics for test set
            _, _, test, _ = self.metrics(prediction_ts, train_set_output)
            train_f_score[i, :] = np.transpose(test)

            # Run metrics for cv set
            _, _, test, _ = self.metrics(prediction_cv, cv_set_output)
            cross_validation_f_score[i, :] = np.transpose(test)

        # Print the class accuracy with associated lambda value
        header_f_str = "{head0:^20.20s} | {head1:^20.20s} | {head2:^20.20s} | {head3:^20.20s} | " \
                       "{head4:^20.20s} | {head5:^20.20s} | {head6:^20.20s}"
        data_f_str = "{col0:^20.5f} | {col1:^20.5f} | {col2:^20.5f} | {col3:^20.5f} | " \
                     "{col4:^20.5f} | {col5:^20.5f} | {col6:^20.5f}"

        # Print Header data
        print(header_f_str.format(head0="Lambda Value", head1="Class 0 TS F Score", head2="Class 1 TS F Score",
                                  head3="Class 2 TS F Score", head4="Class 0 CV F Score", head5="Class 1 CV F Score",
                                  head6="Class 2 CV F Score"))

        # Header/Metrics Data separator
        underline_fs = "{:-^125}"
        print(underline_fs.format(''))

        # Print Lambda/Metrics
        for k in range(ll):
            print(data_f_str.format(col0=(lambda_vector[k])[0], col1=train_f_score[k, 0], col2=train_f_score[k, 1],
                                    col3=train_f_score[k, 2], col4=cross_validation_f_score[k, 0],
                                    col5=cross_validation_f_score[k, 1], col6=cross_validation_f_score[k, 2]))

        """    
        # Plot the training error and validation error vs value of lambda
        plt.plot(train_error, lambda_vector, 'r')
        plt.plot(cross_validation_error, lambda_vector, 'b')
        plt.title("Training Error, Cross Validation Error vs. Lambda Values")
        plt.table([["Training Error"], ["Cross Validation Error"]], [['r'], ['b']])
        plt.grid(True, 'both', 'both')
        plt.xlabel("Lambda Value")
        plt.ylabel("Cost")

        # Find the lambda value associated with the lowest cv error
        optimal_lambda = lambda_vector[0]
        temp_cv_error = cross_validation_error[0]
        for j in range(1, ll):
            if cross_validation_error[j] < temp_cv_error:
                optimal_lambda = lambda_vector[j]
                temp_cv_error = cross_validation_error[j]
        """
        return 0#optimal_lambda

    """
    Name       :
    Purpose    : 
    Parameters :
    Return     :
    Notes      :
    """
    def metrics(self, predicted_output, actual_output):
        # Precision, Recall, Accuracy, F_Score array initialization
        precision_arr = np.zeros((self.num_classes, 1))
        recall_arr = np.zeros((self.num_classes, 1))
        accuracy_arr = np.zeros((self.num_classes, 1))
        f_score_arr = np.zeros((self.num_classes, 1))

        # Find precision and recall for each class
        for i in range(self.num_classes):
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            for j in range(len(predicted_output)):
                # Tabulate tp, fp, fn for class i
                if predicted_output[j] == i and actual_output[j] == i:
                    tp += 1
                elif predicted_output[j] == i and actual_output[j] != i:
                    fp += 1
                elif predicted_output[j] != i and actual_output[j] == i:
                    fn += 1
                elif predicted_output[j] != i and actual_output[j] != i and (predicted_output[j] == actual_output[j]):
                    tn += 1

            # Calculate precision and recall for class i
            try:
                precision_arr[i] = tp / (tp + fp)
            except ZeroDivisionError:
                print("Divide by zero mess up!")
                precision_arr[i] = 0
            try:
                recall_arr[i] = tp / (tp + fn)
            except ZeroDivisionError:
                print("Divide by zero mess up!")
                recall_arr[i] = 0

            # Calculate Accuracy for the given class
            accuracy_arr[i] = (tp + tn) / (tp + fp + fn + tn)

            # Calculate F_score for the given class
            f_score_arr[i] = (2 * precision_arr[i] * recall_arr[i]) / (precision_arr[i] + recall_arr[i])

        """
        # Set data format strings
        header_f_str = "{head0:^20s} | {head1:^20s} | {head2:^20s} | {head3:^20s} | {head4:^20s}"
        class_f_str = "{col0:^20.25s} | {col1:^20.4f} | {col2:^20.4f} | {col3:^20.4f} | {col4:^20.4f}"

        # Print the Header data
        print(header_f_str.format(head0="Class Name", head1="Precision", head2="Recall", head3="F Score",
                                  head4="Accuracy"))
        # Header/Metrics Data separator
        underline_fs = "{:-^125}"
        print(underline_fs.format(''))

        # Print Metrics Data
        for j in range(self.num_classes):
            class_name = "Class " + str(j)
            print(class_f_str.format(col0=class_name, col1=precision_arr[j][0], col2=recall_arr[j][0],
                                     col3=f_score_arr[j][0], col4=accuracy_arr[j][0]))
        """

        return precision_arr, recall_arr, f_score_arr, accuracy_arr

    """
    Name       : store_parameters
    Purpose    : To write the learned parameter values to a csv file
    Parameters : theta_values which is a matrix of learned parameter values
    Return     : None
    Notes      : 
        This function truncates any data already in the csv file
        theta_values has dimensions: (number of classes x number of features)    
    """
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


def main():
    test = LearningAlgorithm()
    test.top_learning_algorithm()

    """
    test.num_classes = 3

    predicted_arr =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    actual_arr = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 1, 1, 0, 1, 1, 2, 2, 2, 2, 2, 2]
    predicted_arr = np.array(predicted_arr)
    actual_arr = np.array(actual_arr)
    predicted_arr = np.reshape(predicted_arr, (25, 1))
    actual_arr = np.reshape(actual_arr, (25, 1))


    test.metrics(predicted_arr, actual_arr)
    """
    return

if __name__ == "__main__":
    main()