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
        self.class_ht = None
        self.class_data_name = "class_data.csv"
        self.total_number_of_examples = 0
        self.csv_data_base_name = 'input_and_output_data_test.csv'
        self.show_images = False


    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
    '''
    def top_data_collection(self):
        # Load in class data
        self.read_class_data()

        # Prompt user for the file name
        file_name, file_extension = self.file_name_prompt()

        # Prompt the user for batch/individual and class number
        class_num = self.batch_vs_individual_prompt()

        # Partition the image(s) and write data to csv
        self.partition_image_top(class_num, file_name, file_extension)

        # Display post data collection class data
        self.display_class_data()

        # Write updated class data to csv
        self.write_class_data()

        return

    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
    '''
    def partition_image_top(self, class_number, file_name, file_type):
        if class_number == -1:
            self.partition_image((file_name + file_type), class_number, self.total_number_of_examples)
        else:
            try:
                file_name_idx = 0
                while True:
                    full_file_name = file_name + str(file_name_idx) + file_type
                    self.partition_image(full_file_name, class_number, self.total_number_of_examples)
                    file_name_idx += 1
            except IOError:
                print("Max number of images reached")
                return
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

        # Show images
        if self.show_images:
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

                # Class assignment
                class_assign = None
                if class_index == -1:
                    class_assign = self.individually_assign_class_prompt(pixel_box_rgb)
                else:
                    class_assign = class_index

                # Update class data
                class_name, num_class_examples = self.class_ht[class_assign]
                self.class_ht[class_assign] = (class_name, num_class_examples + 1)

                # Append class index data
                data_pixel_box_array = np.append(data_pixel_box_array, class_assign)

                # Write pixel data t
                self.write_data_to_data_base(data_pixel_box_array, self.csv_data_base_name)

                # Show count
                curr_image_num += 1
                self.total_number_of_examples += 1
                print("Current CSV index: " + str(self.total_number_of_examples))

        print("Done Writing Data")
        return

    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
    '''
    @staticmethod
    def file_name_prompt():
        # Prompt user for number of file inputs
        multiple_files = None
        while multiple_files is None and multiple_files != "Y" and multiple_files != "N" and multiple_files != "y" \
                and multiple_files != "n":
            multiple_files = input("Are you inputting more than one file? (Y/N): ")

        # File name specific prompt for multiple files
        if multiple_files == "Y":
            print("For multiple files ensure the image names are the same with the last character in the string"
                  " being number starting with 0")
            print("\tFor example: file_name_0, file_name_1, file_name_2...\n\tinputFile0, inputFile1, inputFile2,...\n")
            print("For the image extension please include the dot prefix")
            print("\tFor example: .JPG, .PNG\n")

        # Prompt user for file name and double check it is correct
        file_name = None
        file_extension = None
        while True:
            file_name = input("Please enter the image name: \n")
            file_extension = input("Please enter the image extension: \n")
            file_check = input("Is the following information correct? (Y/N) \n\tFile Name: %s\n\tImage Extension: %s\n"
                               % (file_name, file_extension))
            if file_check == "Y" or file_check == 'y':
                break
            else:
                print("Error: Please try again\n")

        return file_name, file_extension

    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
    '''
    def batch_vs_individual_prompt(self):
        # Return value instantiation
        batch_class_num = -1

        # Display Class Data
        self.display_class_data()

        # Prompt for batch/individual and class number
        while True:
            single_class = input("Would you like to assign all sub boxes in the given\n\t"
                                 " picutre(s) to one class? (Y/N): \n")
            if single_class == "Y" or single_class == "y":
                batch_class_num = self.class_number_prompt()
                break
            elif single_class == "N" or single_class == "n":
                break
            else:
                print("Input Error: Try again\n")

        return batch_class_num

    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
    '''
    def class_number_prompt(self):
        user_input_class_number = -1
        while True:
            class_number = int(input("Please enter the class number: \n"))
            if class_number not in self.class_ht.keys():
                new_class_response = input("This class does not exist. Would you like to add a new class? (Y/N) \n")
                if new_class_response == 'Y' or new_class_response == 'y':
                    new_class_name = input("Please enter the new class name: \n")
                    self.class_ht[class_number] = (new_class_name, 0)
                    user_input_class_number = class_number
                    break
                else:
                    print("Input Error: Try again\n")
            else:
                user_input_class_number = class_number
                break

        return user_input_class_number

    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
    '''
    def individually_assign_class_prompt(self, subpixel_box) -> int:
        class_number = -1
        self.display_class_data()
        while class_number == -1:
            subpixel_box.resize((600, 600)).show()
            #self.display_class_data()
            class_number = self.class_number_prompt()
        return class_number

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
            key == class number
            value == (class name:str , number_examples: int)
    '''
    def read_class_data(self):
        try:
            with open(self.class_data_name) as class_data_csv:
                reader = csv.reader(class_data_csv)
                data_list = list(reader)

            # Dictionary Data Conversion
            class_data_ht = {}
            for class_num, class_name, num_class_examples in data_list:
                class_data_ht[int(class_num)] = (class_name, int(num_class_examples))

            # Store class data in ht field
            self.class_ht = class_data_ht

            # Count the total number of examples processed
            temp_total = 0
            for key, values in self.class_ht.items():
                temp_total += values[1]
            self.total_number_of_examples = temp_total

        except FileNotFoundError:
            print("No class data file found.\n")
            self.class_ht = {}
        except ValueError:
            print("Empty class data file\n")
            self.class_ht = {}

        return

    '''
    Name       : 
    Purpose    : 
    Parameters : 
    Return     : 
    Notes      :
            key == class number
            value == (class name:str , number_examples: int)
    '''
    def write_class_data(self):
        file_name = self.class_data_name

        # Write class data dictionary
        try:
            with open(file_name, 'w+', newline='') as class_data_file:
                writer = csv.writer(class_data_file)
                for key, values in self.class_ht.items():
                    writer.writerow([key, values[0], values[1]])
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
    def display_class_data(self):
        print("The current class data is as follows: ")
        print("Class\tClass_Name\t\t\tNumber of Examples")
        for key, values in self.class_ht.items():
            print("%d    \t%s\t\t%d" % (key, values[0], values[1]))
        print("\n")
        return

'''
Name       : 
Purpose    : 
Parameters : 
Return     : 
Notes      :
'''


def main():
    obj_data_collection = DataCollection()
    obj_data_collection.top_data_collection()
    return

if __name__ == '__main__':
    main()