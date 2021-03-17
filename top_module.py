from new_image_prediction import *
from learning_algorithm import *
from data_creation import *


def main():
    obj_learn_algo = LearningAlgorithm()
    obj_learn_algo.top_learning_algorithm()

    obj_new_img_pred = ImagePrediction(obj_learn_algo.num_classes)

    return

if __name__ == "__main__":
    main()
