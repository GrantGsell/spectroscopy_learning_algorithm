import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image as img
import matplotlib as mpl
#mpl.use('TkAgg')
plt.ioff()


class ImagePrediction:

    def __init__(self, auto_flag):
        self.num_classes = None
        self.parameter_file_name = 'parameter_values.csv'
        self.num_row_pixels = 40
        self.num_col_pixels = 40
        self.img_height = None
        self.img_width = None
        self.auto_flag = auto_flag

    '''
    Name       :
    Purpose    : 
    Parameters :
    Return     :
    Notes      :
    '''
    def top_image_prediction(self, new_image_name):
        # Read in the parameter data
        parameters = self.read_parameters()

        # Partition and Prediction for new image
        image_ht = self.new_image_partition_and_prediction(new_image_name, parameters, self.auto_flag)

        # Convert Prediction into a binary matrix
        binary_prediction_matrix = self.create_binary_matrix(image_ht, self.img_height // self.num_row_pixels, self.img_width // self.num_col_pixels)

        # Write the binary matrix data
        self.write_binary_matrix(binary_prediction_matrix)

        return




    '''
    Name       :
    Purpose    : 
    Parameters :
    Return     :
    Notes      :
    '''
    def read_parameters(self):
        data = np.loadtxt(open(self.parameter_file_name), delimiter=",", dtype='float32')
        parameters = data[:, :]
        self.num_classes = (np.shape(parameters))[0]
        return parameters


    '''
    Name       :
    Purpose    : 
    Parameters :
    Return     :
    Notes      :
    '''
    def new_image_partition_and_prediction(self, file_name: str, parameter_values, auto_flag: bool):
        # Create a dictionary to hold the image data
        image_ht = dict()

        # Read in the new image
        new_image = img.open(file_name)
        new_image.load()

        # Display the image if not set to automatic
        if not auto_flag:
            new_image.show()

        # Obtain image height, width
        height = new_image.height
        self.img_height = new_image.height
        width = new_image.width
        self.img_width = new_image.width

        # Partition the image into m x n pixel boxes
        m = self.num_row_pixels
        n = self.num_col_pixels

        # Create a figure with (width / m) x (height / n) subplots
        if not auto_flag:
            fig, ax = plt.subplots((height // n), (width // m), figsize=(16, 12), gridspec_kw={'wspace': 0.20, 'hspace': 0.20},
                                   subplot_kw={'xticklabels': [], 'yticklabels': [], 'xticks': [], 'yticks': [], 'picker': True})

        # Partition original image into boxes and analyze
        for row in range(m, height + 1, m):
            for col in range(n, width + 1, n):
                # Pixel Box Boundaries
                left, right, upper, lower = col - n, col, row - m, row
                box = (left, upper, right, lower)

                # Partition, Show Partition and Obtain Data
                pixel_box = new_image.crop(box)

                # Obtain the RGB pixel data
                data_pixel_box = list(pixel_box.getdata())

                # Turn RGB data into a singular array
                data_pixel_box_array = np.array([elem for tuples in data_pixel_box for elem in tuples])
                data_pixel_box_array = np.reshape(data_pixel_box_array, (1, np.shape(data_pixel_box_array)[0]))

                # Make the prediction for this particular pixel subbox
                curr_prediction = self.predict_one_vs_all(parameter_values, data_pixel_box_array)
                curr_prediction = int(curr_prediction[0])

                # Add sub-box to hashtable
                data_pixel_box_array = np.append(data_pixel_box_array, curr_prediction)
                image_ht[(row // m - 1, col // n - 1)] = data_pixel_box_array

                # Plot the sub box image if auto_flag is false
                if not auto_flag:
                    ax[row // m - 1][col // n - 1].imshow(pixel_box)
                    ax[row // m - 1][col // n - 1].set_aspect('equal')
                    ax[row // m - 1][col // n - 1].set_title(str(curr_prediction), y=-0.1)

                    # Add Column markers to the data if in the first row
                    if(row // m - 1) == 0:
                        ax[row // m - 1][col // n - 1].set_xlabel(str(col // m - 1))
                        ax[row // m - 1][col // n - 1].xaxis.set_label_position('top')

                    # Add Row Markers to the data if in the first column
                    if(col // n - 1) == 0:
                        ax[row // m - 1][col // n - 1].set_ylabel(str(row // m - 1))

        if not auto_flag:
            # Show image post-classification
            fig.show()

        # Manually Change Prediction
        if not auto_flag:
            image_ht = self.manually_change_prediction(image_ht, ax, height // n - 1, width // m - 1, self.num_classes)
            fig.show()

        return image_ht


    '''
    Name       :
    Purpose    :
    Parameters :
    Return     :
    Notes      :
    '''
    def manually_change_prediction(self, prediction_dict: dict, ax, max_row, max_col, num_classes):
        # Prompt user for value changes
        while True:
            change_values = input('Would you like to change any of the predictions? (Y/n) ')
            if change_values == "Y": break
            elif change_values == "n": return prediction_dict
            else: print("Error: Incorrect Input")

        # Prompt users for which sub-box to change
        change_flag = True
        sub_box_hm = dict()
        while change_flag:
            curr_row = -1
            curr_col = -1
            new_val = -1

            # Obtain row number
            while True:
                curr_row = int(input("Enter the row number of the box you want to change: "))
                if curr_row >= 0 and curr_row <= max_row: break
                else: print("Error: Incorrect Input")

            # Obtain column number
            while True:
                curr_col = int(input("Enter the column number of the box you want to change: "))
                if curr_col >= 0 and curr_col <= max_col: break
                else: print("Error: Incorrect Input")

            # Obtain new box value
            while True:
                curr_prediction = prediction_dict[(curr_row, curr_col)][-1]
                print("The current prediction class for this sub box is: %d" % curr_prediction)
                new_val = int(input("Enter the new class you want to assign the sub box class: "))
                if new_val >= 0 and (new_val <= num_classes - 1): break
                else: print("Error: Incorrect Input")

            # Add new data to hm
            sub_box_hm[(curr_row, curr_col)] = new_val

            # Prompt the user for another input
            while True:
                try_again = input("Would you like to change any other predictions? (Y/n) ")
                if try_again == 'Y': break
                elif try_again == 'n':
                    change_flag = False
                    break
                else: print("Error: Incorrect Input")

        # Update the Predictions dictionary
        for key in sub_box_hm.keys():
            update_val = sub_box_hm[key]
            prediction_value = prediction_dict.get(key)
            prediction_value[-1] = update_val
            prediction_dict[key] = prediction_value

        # Visually show the prediction changes
        for key in sub_box_hm.keys():
            row = key[0]
            col = key[1]

            # Change box outline color
            ax[row][col].spines['bottom'].set_color('red')
            ax[row][col].spines['top'].set_color('red')
            ax[row][col].spines['left'].set_color('red')
            ax[row][col].spines['right'].set_color('red')

            # Change box outline thickness
            ax[row][col].spines['bottom'].set_linewidth(5)
            ax[row][col].spines['top'].set_linewidth(5)
            ax[row][col].spines['left'].set_linewidth(5)
            ax[row][col].spines['right'].set_linewidth(5)

            new_prediction = prediction_dict[key][-1]
            ax[row][col].set_title(str(new_prediction), y=-0.1)

        return prediction_dict

    '''
    Name       :
    Purpose    :
    Parameters :
    Return     :
    Notes      :
    '''
    @staticmethod
    def create_binary_matrix(prediction_dict, num_row, num_col):
        # Create zeros matrix
        bin_mat = np.zeros((num_row, num_col))

        # Iterate through the prediction dictionary
        for key in prediction_dict.keys():
            if prediction_dict.get(key)[-1] > 1.0:
                row = key[0]
                col = key[1]
                bin_mat[row, col] = 1

        return bin_mat

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
    @staticmethod
    def sigmoid_hypothesis(theta, X):
        # Perform matrix multiplication to obtain the sigmoid input vector
        z = np.matmul(X, theta)

        # Perform sigmoid calculations
        g = 1.0 / (1.0 + np.exp(-z))

        return g

    '''
    Name       :
    Purpose    : 
    Parameters :
    Return     :
    Notes      :
    '''
    @staticmethod
    def write_binary_matrix(binary_matrix):
        file_name = 'imaging_matrix.csv'

        # Open the current file and truncate the data
        imaging_matrix_file = open(file_name, 'w+', newline='')

        # Write parameter values
        with imaging_matrix_file:
            writer = csv.writer(imaging_matrix_file)
            writer.writerows(binary_matrix)

        # Close the parameter file
        imaging_matrix_file.close()

        return


def main():
    new_prediction = ImagePrediction(3)
    new_prediction.top_image_prediction('new_image_test.jpg', False)

    return

if __name__ == '__main__': main()