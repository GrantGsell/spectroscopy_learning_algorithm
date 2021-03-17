from PIL import Image as img
import csv
import numpy as np

'''
File Name   :
Author      :
Date        :
Description :
'''


'''
Name       : 
Purpose    : 
Parameters : 
Return     : The number of examples data has been collected for
Notes      :
'''


def partition_image(file_name: str, class_index: int, total_count: int):
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

            # TEST for writing all data
            write_data_to_data_base(data_pixel_box_array, 'input_and_output_data.csv')

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


def write_data_to_data_base(pixel_data: list, csv_file_name: str) -> None:
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


def assign_class(subpixel_box) -> int:
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
'''


def main():
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
    return 1


if __name__ == '__main__': main()