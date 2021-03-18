from PIL import Image as img
import csv
import numpy as np

'''
File Name   :
Author      :
Date        :
Description :
'''

class DataCollection:

    def __init__(self):
        self.num_examples = None
        self.num_examples_per_class = None
        self.class_ht = None
        self.class_data_name = "class_data.csv"


    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
    '''
    def top_data_collection(self):
        # Prompt user for bulk data colllection or sample by sample collection

        # Prompt the user for the file name

        return

    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : The number of examples data has been collected for
    Notes      :
    '''
    def partition_image(self, file_name: str, class_index: int, total_count: int):
        # Read in the RGB image and display it
        image_rgb = img.open(file_name)
        image_rgb.load()                # PIL 'forgot' to load after opening
        image_rgb.show()

        r, g, b = image_rgb.getpixel((1, 1))

        # Obtain Image height, width == 1200, 1600
        height = image_rgb.height
        width = image_rgb.width

        # Partition the image into m x n pixel boxes
        m = 40
        n = 40

        curr_image_num = 0
        for row in range(m, height + 1, m):
            for col in range(n, width + 1, n):
                # Pixel Box Boundaries
                left, right, upper, lower = col - n, col, row - m, row
                box = (left, upper, right, lower)

                box_size = (right - left) * (lower - upper)
                if box_size != (m * n):
                    print('Box Size Error: %d' % box_size)

                # Partition, Show Partition and Obtain Data
                pixel_box_rgb = image_rgb.crop(box)

                # Write the input and output data to respective files
                data_pixel_box = list(pixel_box_rgb.getdata())
                data_pixel_box_array = np.array([elem for tuples in data_pixel_box for elem in tuples])
                data_pixel_box_array = np.append(data_pixel_box_array, class_index)

                # For writing all data
                self.write_data_to_data_base(data_pixel_box_array, 'input_and_output_data.csv')

                # Show count
                curr_image_num += 1
                total_count += 1
                print("Current CSV index: " + str(total_count))

        print("Done Writing Data")
        return curr_image_num, total_count


    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
    '''
    def file_name_and_class_prompt(self):
        # Prompt user for number of file inputs
        multiple_files = None
        while multiple_files is None or multiple_files != "Y" or multiple_files == "N":
            multiple_files = input("Are you inputting more than one file? (Y/N): ")

        # File name specific prompt for multiple files
        if multiple_files == "Y":
            print("For multiple files ensure the file names are the same with the last character in the string"
                  " being number.")
            print("\tFor example: file_name_1, file_name_2...\ninputFile1, inputFile2,...")

        # Prompt user for file name and double check it is correct
        file_name = None
        file_type = None
        while True:
            file_name = input("Please enter the file name: ")
            file_type = input("\nPlease enter the file type: ")
            file_check = input("Is the following information correct? (Y/N) \n\tFile Name: %s\n\tFile Type: %s"
                               % (file_name, file_type))
            if file_check == "Y" or file_check == 'y':
                break
            else:
                print("Error: Please try again\n")

        # Prompt user for class number (if new prompt for new name too)
        while True:
            break

        return file_name, file_type

    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
    '''
    def write_data_to_data_base(self, pixel_data: list, csv_file_name: str) -> None:
        with open(csv_file_name, mode='a') as storage:
            data_writer = csv.writer(storage, delimiter=',', lineterminator='\n')
            data_writer.writerow(pixel_data)
        return


    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
    '''
    def assign_class(self, subpixel_box) -> int:
        class_number = -1
        class_map = {0, 1, 2, 3}
        while class_number == -1:

            subpixel_box.resize((600, 600)).show()
            #print('Class 0 Denotes: Tin Foil\nClass 1 Denotes: Well Plate\nClass 2 Denotes: Well Plate with Object\nClass '
            #      '3 Denotes: Other')
            temp = input('What class does this subpixel box belong to: ')
            if int(temp) in class_map:
                class_number = int(temp)
        return class_number

    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
            key == class number
            value == (class name:str , number_examples: int)
    '''
    def read_class_data(self) -> dict:
        # Read in class data
        data = np.loadtxt(open(self.class_data_name), delimiter=",")
        class_keys = data[:, 0]
        class_data = data[:, 1:]

        # Transform class data into hash table
        class_ht = dict()
        for idx in range(len(class_keys)):
            curr_class_key = class_keys[idx, 0]
            curr_class_name = class_data[idx, 1]
            curr_class_count = int(class_data[idx, 2])
            class_ht[curr_class_key] = (curr_class_name, curr_class_count)

        return class_ht

    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
            key == class number
            value == (class name:str , number_examples: int)
    '''
    def write_class_data(self, class_ht):
        file_name = self.class_data_name

        # Open the current file and truncate the data
        class_data_file = open(file_name, 'w+', newline='')

        # CSV columns field names
        csv_fields = ['Class Number', "Class Name", 'Number of Examples']

        # Write class data dictionary
        try:
            with class_data_file:
                writer = csv.DictWriter(class_ht, fieldnames=csv_fields)
                writer.writeheader()
                for data in class_ht:
                    writer.writerows(data)
        except IOError:
            print("I/O error")

        # Close the class data file
        class_data_file.close()

        return

'''
Name       : 
Purpose    : 
Parameters : 
Return     : 
Notes      :
'''


def main():
    '''
    total_count = 0

    # Class 0 (Tin Foil) Data Collection
    num_class_0_examples = 0
    for l in range(1):
        image_name_0 = "tin_foil_" + str(l) + "_ver_1.jpg"
        num_class_0_examples_temp, total_count = partition_image(image_name_0, 0, total_count)
        num_class_0_examples += num_class_0_examples_temp

    # Class 1 (Well Plate) Data Collection
    num_class_1_examples = 0
    for i in range(1):
        image_name_1 = "well_plate_" + str(i) + "_ver_1.jpg"
        num_class_1_examples_temp, total_count = partition_image(image_name_1, 1, total_count)
        num_class_1_examples += num_class_1_examples_temp

    # Class 2 (Sample Substitution) Data Collection
    num_class_2_examples = 0
    for j in range(1):
        image_name_2 = "cell_sub_" + str(j) + ".jpg"
        num_class_2_examples_temp, total_count = partition_image(image_name_2, 2, total_count)
        num_class_2_examples += num_class_2_examples_temp


    # Report of Number of Examples Taken for each class
    print("Number of examples for each Class: ")
    print("Class 0: " + str(num_class_0_examples))
    print("Class 1: " + str(num_class_1_examples))
    print("Class 2: " + str(num_class_2_examples))
    print("Main Function Done")
    '''

    return

if __name__ == '__main__':
    main()