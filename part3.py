import matplotlib.pyplot as plt

def main():
    x = [1,2,3,4,5,6,7,8,9,10]
    training_y_small = [0.94, 0.88, 1, 1, 1, 0.995, 1, 0.995, 1, 1]
    test_y_small = [0.68, 0.555, 0.625, 0.625, 0.71, 0.62, 0.69, 0.68, 0.705, 0.67]
    training_y_FDDB = [0.81, 0.81, 0.856, 0.856, 0.862, 0.872, 0.833, 0.75, 0.561, 0.55]
    test_y_FDDB = [0.78, 0.78, 0.825, 0.832, 0.845, 0.826, 0.796, 0.729, 0.567, 0.555]
    
    # plt.plot(x, training_y_small, 'o-')
    # plt.plot(x, test_y_small, 'o-')
    plt.plot(x, training_y_FDDB, 'o-')
    plt.plot(x, test_y_FDDB, 'o-')
    plt.legend(['training', 'test'])
    plt.title('Accuracy per iteration')
    plt.show()

if __name__ == "__main__":
    main()