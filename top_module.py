from new_image_prediction import *
from learning_algorithm import *
from data_creation import *


def main():
    obj_learn_algo = LearningAlgorithm()
    obj_learn_algo.top_learning_algorithm()

    num_classes = obj_learn_algo.num_classes
    obj_new_img_pred = ImagePrediction(num_classes, False)
    obj_new_img_pred.top_image_prediction('new_image_test.jpg')

    return

if __name__ == "__main__":
    main()
