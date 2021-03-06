import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image as img
import os
import pathlib
import matplotlib as mpl
#mpl.use('TkAgg')
plt.ioff()


class ImagePrediction:

    def __init__(self, auto_flag: bool):
        self.num_classes = None
        self.parameter_file_name = 'parameter_values.csv'
        self.new_file_name = "test_pic_v0.jpg"
        self.num_row_pixels = 40
        self.num_col_pixels = 40
        self.img_height = None
        self.img_width = None
        self.auto_flag = auto_flag

    '''
    Name       : top_image_prediction
    Purpose    : To take a new image, partition it into sub-boxes, make a prediction on each sub-box, store the 
                    predictions in a dictionary, translate the dictionary into a binary matrix and write the binary
                    matrix to a csv file.
    Parameters : None
    Return     : None
    Notes      :
                 This method assumes the new image is in the cwd
    '''
    def top_image_prediction(self, new_image_name):
        # Take a new picture
        self.camera_capture(True)

        # Read in the parameter data
        parameters = self.read_parameters()

        # Partition and Prediction for new image
        image_ht = self.new_image_partition_and_prediction(self.new_file_name, parameters, self.auto_flag)

        # Convert Prediction into a binary matrix
        binary_prediction_matrix = self.create_binary_matrix(image_ht,
                        self.img_height // self.num_row_pixels, self.img_width // self.num_col_pixels)

        # Write the binary matrix data
        self.write_binary_matrix(binary_prediction_matrix)

        return

    '''
    Name       : new_image_partition_and_prediction
    Purpose    : To partition the new image into sub-boxes and make new predictions on each sub-box.
    Parameters :
                 file_name, a string denoting the name of the new image to be processed.
                 parameter_values, a ndarray denoting the learned parameters for the algorithm.
                 auto_flag, a boolean that denotes whether or not the user wants the ability to change any of the
                    predictions made on each sub-box. If true the user will be prompted to change any of the predicted
                    values, if false no changes can be made to the sub-box prediction.
    Return     : image_ht, a dictionary containing the prediction for each sub-box
    Notes      :
                 image_ht dictionary: {
                                        key: tuple = (sub-box row, sub-box column) 
                                        value: ndarray = [pixel sub-box data :  sub-box prediction]
                                       }
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
            fig, ax = plt.subplots((height // n), (width // m), figsize=(16, 12),
                                    gridspec_kw={'wspace': 0.20, 'hspace': 0.20},
                                    subplot_kw={'xticklabels': [], 'yticklabels': [], 'xticks': [], 'yticks': [],
                                                'picker': True})

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
    Name       : manually_change_prediction
    Purpose    : Allows the user to change any of the sub-box predictions made by the algorithm.
    Parameters : 
                 prediction_dict which is a dictionary denoting the prediction made by the algorithm for each sub-box.
                 ax which is the axis attribute of the image plot.
                 max_row which is an integer denoting the maximum number of sub-box rows.
                 max_col which is an integer denoting the maximum number of sub-box columns.
                 num_classes which is an integer denoting the number of classes.
    Return     : prediction_dict which is a dictionary denoting the prediction made by the algorithm for each sub-box.
    Notes      :
                 Changes are based on matrix row/col input values.
                 Predictions can only be changed to pre-existing classes.
    '''
    def manually_change_prediction(self, prediction_dict: dict, ax, max_row, max_col, num_classes) -> dict:
        # Prompt user for value changes
        while True:
            change_values = input('Would you like to change any of the predictions? (Y/n) ')
            if change_values == "Y" or change_values == 'y':
                break
            elif change_values == "N" or change_values == 'n':
                return prediction_dict
            else:
                print("Error: Incorrect Input")

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
                if 0 <= curr_row <= max_row:
                    break
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
    Name       : create_binary_matrix
    Purpose    : To translate the prediction dictionary into a 0/1 matrix.
    Parameters : 
                 prediction_dict which is a dictionary denoting the prediction for each sub-box.
                 num_row which is an integer denoting the maximum number of sub-box rows.
                 num_col which is an integer denoting the maximum number of sub-box columns.
    Return     : binary_matrix which is a ndarray.
    Notes      :
                 A 0 denotes a sub-box that should not be imaged.
                 A 1 denotes a sub-box that should be imaged.
                 Classes 0/1 are assigned a value of 0.
                 All other classes are assigned a value of 1.
    '''
    @staticmethod
    def create_binary_matrix(prediction_dict, num_row, num_col):
        # Create zeros matrix
        binary_matrix = np.zeros((num_row, num_col))

        # Iterate through the prediction dictionary
        for key in prediction_dict.keys():
            if prediction_dict.get(key)[-1] > 1.0:
                row = key[0]
                col = key[1]
                binary_matrix[row, col] = 1

        return binary_matrix

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

    """
    Name       : write_binary_matrix
    Purpose    : Writes the binary matrix for the image taken to a csv file.
    Parameters : binary_matrix, a ndarray
    Return     : None
    Notes      : None
    """
    @staticmethod
    def write_binary_matrix(binary_matrix) -> None:
        file_name = "imaging_matrix.csv"

        # Open the current file and truncate the data
        imaging_matrix_file = open(file_name, "w+", newline="")

        # Write parameter values
        with imaging_matrix_file:
            writer = csv.writer(imaging_matrix_file)
            writer.writerows(binary_matrix)

        # Close the parameter file
        imaging_matrix_file.close()

        return

    '''
    Name       : read_parameters
    Purpose    : To read in the learned parameters.
    Parameters : None
    Return     : parameters, a ndarray containing the learned parameters.
    Notes      : None
    '''
    def read_parameters(self) -> np.ndarray:
        parameters = None
        try:
            data = np.loadtxt(open(self.parameter_file_name), delimiter=",", dtype="float32")
            parameters = data[:, :]
            self.num_classes = (np.shape(parameters))[0]
        except FileNotFoundError:
            print("The algorithm has not been run yet, please do so.")

        return parameters

    '''
    Name       : camera_capture
    Purpose    : To capture a new image for processing.
    Parameters : preview, a boolean denoting if a preview of the image should be displayed.
    Return     : None
    Notes      :
                 If preview is True, a preview of the image is shown.
                 The camera used is a Raspberry Pi Camera V2.
                 Image Dimensions are set to (1600 x 1200).
                 The image is stored in the cwd.
                 The image takes 500ms to be captured (minimum based on camera specs).
    '''
    @staticmethod
    def camera_capture(preview: bool) -> None:
        # Obtain the cwd and set image path
        cwd = str(pathlib.Path().absolute())
        output_file_name = cwd + "\\test_pic_v0.jpg"

        # Set the command string
        command_string = None
        if preview:
            command_string = "raspistill -f -w 1600 -h 1200 -t 500 -o "
        else:
            command_string = "raspistill -n -w 1600 -h 1200 -t 500 -o "

        # Take the picture and store it
        os.system(command_string + output_file_name)

        return


def main():
    new_prediction = ImagePrediction(False)
    new_prediction.num_classes = 3
    new_prediction.top_image_prediction('new_image_test.jpg')

    return


if __name__ == '__main__':
    main()