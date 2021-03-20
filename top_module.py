from new_image_prediction import *
from learning_algorithm import *
from data_collection import *


class TopModule:

    """
    Name       :
    Purpose    :
    Parameters :
    Return     :
    Notes      :
    """
    @staticmethod
    def top_module_prompt():
        while True:
            user_inp = input("Please enter a number corresponding to one of the following actions: \n"
                             "\t1 : Perform data collection\n"
                             "\t2 : Run the learning algorithm\n"
                             "\t3 : Classify a new image\n"
                             "\t4 : Quit the program\n")
            try:
                user_inp = int(user_inp)
                if user_inp == 1:
                    obj_data_collection = DataCollection()
                    obj_data_collection.top_data_collection()
                elif user_inp == 2:
                    obj_learn_algo = LearningAlgorithm()
                    obj_learn_algo.top_learning_algorithm()
                elif user_inp == 3:
                    obj_new_img_pred = ImagePrediction(False)
                    obj_new_img_pred.top_image_prediction('new_image_test.jpg')
                else:
                    break
            except ValueError:
                print("Input Error: Try again.\n")
                continue
        return


def main():
    runner = TopModule()
    runner.top_module_prompt()
    return

if __name__ == "__main__":
    main()
