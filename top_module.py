from new_image_prediction import *
from learning_algorithm import *
from data_collection import *


def main():
    obj_learn_algo = LearningAlgorithm()
    obj_learn_algo.top_learning_algorithm()

    obj_new_img_pred = ImagePrediction(False)
    obj_new_img_pred.top_image_prediction('new_image_test.jpg')
    print("Done")
    return

if __name__ == "__main__":
    main()
