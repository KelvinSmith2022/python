"""
Importing useful python libraries to be used in this project
"""
import math
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import style
import sqlalchemy
import sqlalchemy as db
import pymysql


def main():
    # Getting access to train csv file and assigning each column to a variable
    train_data = pd.read_csv(
        filepath_or_buffer="C:\\Users\\user\\PycharmProjects\\python_assignment\\datasets\\train.csv")
    x = train_data.iloc[:, 0].values
    y1 = train_data.iloc[:, 1].values
    """
    Plotting all train data values (y1, y2, y3, y4)
    """
    style.use("ggplot")
    plt.plot(x, y1, label="train data y1", linewidth=2)
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('train data y1 Plot')
    plt.show()

    y2 = train_data.iloc[:, 2].values
    style.use("ggplot")
    plt.plot(x, y2, label="train data y2", linewidth=2)
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('train data y2 Plot')
    plt.show()

    y3 = train_data.iloc[:, 3].values
    style.use("ggplot")
    plt.plot(x, y3, label="train data y3", linewidth=2)
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('train data y3 Plot')
    plt.show()

    y4 = train_data.iloc[:, 4].values
    style.use("seaborn-dark")
    plt.plot(x, y4, label="train data y4", linewidth=2)
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('train data y4 Plot')
    plt.show()

    # Getting access to ideal csv file and calculating sum of least squares for each train data column
    ideal_func = pd.read_csv(
        filepath_or_buffer="C:\\Users\\user\\PycharmProjects\\python_assignment\\datasets\\ideal.csv")
    # performing least square on ideal functions
    print("Sum of Least squares for values in columns of ideal csv file")
    """
    Using for loop to iterate through ideal csv file and also declaring an empty array to store sum of least squares 
    and print out the minimum among them
    """
    i = 0
    best_fit1 = []
    for columns in ideal_func:
        ideal_func1 = ideal_func.iloc[:, i].values
        least_square1 = np.sum((y1 - ideal_func1) ** 2)
        print(columns + " :" + str(least_square1))
        best_fit1.append(least_square1)
        i = i + 1
    print(min(best_fit1))
    print("Best fit ideal function for train data(y1) is y17")
    print()
    print("Sum of Least squares for y2 values")
    k = 0
    best_fit2 = []
    for columns in ideal_func:
        ideal_func2 = ideal_func.iloc[:, k].values
        least_square2 = np.sum((y2 - ideal_func2) ** 2)
        print(columns + " :" + str(least_square2))
        best_fit2.append(least_square2)
        k = k + 1
    print(min(best_fit2))
    print("Best fit ideal function for train data(y2) is y33")
    print()
    print("Sum of Least square for y3 values")
    l = 0
    best_fit3 = []
    for columns in ideal_func:
        ideal_func3 = ideal_func.iloc[:, l].values
        least_square3 = np.sum((y3 - ideal_func3) ** 2)
        print(columns + " :" + str(least_square3))
        best_fit3.append(least_square3)
        l = l + 1
    print(min(best_fit3))
    print("Best fit ideal function for train data(y3) is y39")
    print()
    print("Sum of Least square for y4 values")
    m = 0
    best_fit4 = []
    for columns in ideal_func:
        ideal_func4 = ideal_func.iloc[:, m].values
        least_square4 = np.sum((y4 - ideal_func4) ** 2)
        print(columns + " : " + str(least_square4))
        best_fit4.append(least_square4)
        m = m + 1
    print(min(best_fit4))
    print("Best fit ideal function for train data(y4) is y47")
    print()
    print()

    # Getting access to test csv file and assigning each column to a variable
    test_data = pd.read_csv(
        filepath_or_buffer="C:\\Users\\user\\PycharmProjects\\python_assignment\\datasets\\test.csv")
    x1 = test_data.iloc[:, 0].values
    y1 = test_data.iloc[:, 1].values

    style.use("bmh")
    plt.plot(x1, label="test data x1", linewidth=2)
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('test data x1 Plot')
    plt.show()

    style.use("bmh")
    plt.plot(y1, label="test data y1", linewidth=2)
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('test data y1 Plot')
    plt.show()

    best_func1 = ideal_func.iloc[:, 17].values
    best_func2 = ideal_func.iloc[:, 33].values
    best_func3 = ideal_func.iloc[:, 39].values
    best_func4 = ideal_func.iloc[:, 47].values

    style.use("seaborn-whitegrid")
    plt.plot(best_func1, label="ideal function y17", linewidth=2)
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('Best fit functions for train data  Y1 Plot')
    plt.show()

    style.use("seaborn-whitegrid")
    plt.plot(best_func2, label="ideal function y33", linewidth=2)
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('Best fit functions for train data Y2 Plot')
    plt.show()

    style.use("seaborn-whitegrid")
    plt.plot(best_func3, label="ideal function y39", linewidth=2)
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('Best fit functions for train data Y3 Plot')
    plt.show()

    style.use("seaborn-whitegrid")
    plt.plot(best_func4, label="ideal function y47", linewidth=2)
    plt.grid(True, color="k")
    plt.ylabel('y axis')
    plt.xlabel('x axis')
    plt.title('Best fit functions for train data Y4 Plot')
    plt.show()

    # calculating The deviation for x column in test csv file
    x_mean = np.mean(x1, axis=None)
    print("Mean for x is : " + str(x_mean))
    # Deviation is the value minus the mean of set of values
    dev_test = x1 - x_mean
    print(dev_test)
    print()

    # Calculating the deviation of ideal function y17
    best_func1_mean = np.mean(best_func1, axis=None)
    print("Mean for y17 in ideal function : " + str(best_func1_mean))
    dev_best_func1 = (best_func1 - best_func1_mean)
    print()

    # Calculating the deviation of train y1
    y1_train_mean = np.mean(y1, axis=None)
    print("Mean for y1 in train data is : " + str(y1_train_mean))
    dev_y1_train = (y1 - y1_train_mean)
    print()

    # Calculating the deviation of ideal function y33
    best_func2_mean = np.mean(best_func2, axis=None)
    print("Mean for y17 in ideal function : " + str(best_func2_mean))
    dev_best_func2 = (best_func2 - best_func2_mean)
    print()

    # Calculating the deviation of train y2
    y2_train_mean = np.mean(y2, axis=None)
    print("Mean for y2 in train data is : " + str(y2_train_mean))
    dev_y2_train = (y2 - y2_train_mean)
    print()

    # Calculating the deviation of ideal function y3
    best_func3_mean = np.mean(best_func3, axis=None)
    print("Mean for y39 in ideal function : " + str(best_func3_mean))
    dev_best_func3 = (best_func3 - best_func3_mean)
    print()

    # Calculating the deviation of train y3
    y3_train_mean = np.mean(y3, axis=None)
    print("Mean for y3 in train data is : " + str(y3_train_mean))
    dev_y3_train = (y3 - y3_train_mean)
    print()

    # Calculating the deviation of ideal function y47
    best_func4_mean = np.mean(best_func4, axis=None)
    print("Mean for y47 in ideal function : " + str(best_func4_mean))
    dev_best_func4 = (best_func4 - best_func4_mean)
    print()

    # Calculating the deviation of train y1
    y4_train_mean = np.mean(y4, axis=None)
    print("Mean for y4 in train data is : " + str(y4_train_mean))
    dev_y4_train = (y4 - y4_train_mean)
    print()

    # computing maximum deviation between train and its ideal function times sqrt of 2
    print("This is for y1")
    dataset1 = []
    i = 0
    for row in dev_y1_train:

        c1 = max(row, best_func1[i]) * math.sqrt(2)
        if dev_test[i] < c1:
            dev_best_func1[i] = x1[i]
            print(dev_best_func1[i])
            dataset1.append(dev_best_func1[i])

        else:
            print("Cannot be mapped")
            dev_best_func1[i] = best_func1[i]
            dataset1.append(dev_best_func1[i])
        i = i + 1
    print()
    print("This is for y2")
    dataset2 = []
    i = 0
    for row in range(100):

        c1 = max(dev_y2_train[i], best_func2[i]) * math.sqrt(2)
        if dev_test[i] < c1:
            dev_best_func2[i] = x1[i]
            print(dev_best_func2[i])
            dataset2.append(dev_best_func2[i])

        else:
            print("Cannot be mapped")
            dev_best_func2[i] = best_func2[i]
            dataset2.append(dev_best_func2[i])
        i = i + 1
    print()
    print("This is for y3")
    dataset3 = []
    i = 0
    for row in range(100):

        c1 = max(dev_y3_train[i], best_func3[i]) * math.sqrt(2)
        if dev_test[i] < c1:
            dev_best_func3[i] = x1[i]
            print(dev_best_func3[i])
            dataset3.append(dev_best_func3[i])

        else:
            print("Cannot be mapped")
            dev_best_func3[i] = best_func3[i]
            dataset3.append(dev_best_func3[i])
        i = i + 1
    print()
    print("This is for y4")
    dataset4 = []
    i = 0
    for row in range(100):

        c1 = max(dev_y4_train[i], best_func4[i]) * math.sqrt(2)
        if dev_test[i] < c1:
            dev_best_func4[i] = x1[i]
            print(dev_best_func4[i])
            dataset4.append(dev_best_func4[i])
        else:
            print("Cannot be mapped")
            dev_best_func4[i] = best_func4[i]
            dataset4.append(dev_best_func4[i])
        i = i + 1
    print("Mapping for test data with y17 ideal function : " + str(dataset1))
    print("Mapping for test data with y33 ideal function : " + str(dataset2))
    print("Mapping for test data with y39 ideal function : " + str(dataset3))
    print("Mapping for test data with y47 ideal function : " + str(dataset4))


def details_train():
    engine = db.create_engine("mysql+pymysql://root:$Firmino11@localhost/details")
    connection = engine.connect()
    meta_data = db.MetaData()
    train = db.Table(
        "train", meta_data,
        db.Column("x", db.Float, nullable=False),
        db.Column("y1", db.Float, nullable=False),
        db.Column("y2", db.Float, nullable=False),
        db.Column("y3", db.Float, nullable=False),
        db.Column("y4", db.Float, nullable=False)
    )
    meta_data.create_all(engine)

    # Inserting into created table
    train_data = pd.read_csv(
        filepath_or_buffer="C:\\Users\\user\\PycharmProjects\\python_assignment\\datasets\\train.csv")
    df = pd.DataFrame(train_data)
    train_table = db.Table("train", meta_data, autoload=True, autoload_with=engine)
    for i, row in train_data.iterrows():
        sql_query = "INSERT INTO details.train VALUES (%s,%s,%s,%s,%s)"
        connection.execute(sql_query, row)
        print("Items inserted successfully")


def details_ideal():
    engine = db.create_engine("mysql+pymysql://root:$Firmino11@localhost/details")
    connection = engine.connect()
    meta_data = db.MetaData()
    ideal = db.Table(
        "ideal", meta_data,
        db.Column("x", db.Float, nullable=False),
        db.Column("y1", db.Float, nullable=False),
        db.Column("y2", db.Float, nullable=False),
        db.Column("y3", db.Float, nullable=False),
        db.Column("y4", db.Float, nullable=False),
        db.Column("y5", db.Float, nullable=False),
        db.Column("y6", db.Float, nullable=False),
        db.Column("y7", db.Float, nullable=False),
        db.Column("y8", db.Float, nullable=False),
        db.Column("y9", db.Float, nullable=False),
        db.Column("y10", db.Float, nullable=False),
        db.Column("y11", db.Float, nullable=False),
        db.Column("y12", db.Float, nullable=False),
        db.Column("y13", db.Float, nullable=False),
        db.Column("y14", db.Float, nullable=False),
        db.Column("y15", db.Float, nullable=False),
        db.Column("y16", db.Float, nullable=False),
        db.Column("y17", db.Float, nullable=False),
        db.Column("y18", db.Float, nullable=False),
        db.Column("y19", db.Float, nullable=False),
        db.Column("y20", db.Float, nullable=False),
        db.Column("y21", db.Float, nullable=False),
        db.Column("y22", db.Float, nullable=False),
        db.Column("y23", db.Float, nullable=False),
        db.Column("y24", db.Float, nullable=False),
        db.Column("y25", db.Float, nullable=False),
        db.Column("y26", db.Float, nullable=False),
        db.Column("y27", db.Float, nullable=False),
        db.Column("y28", db.Float, nullable=False),
        db.Column("y29", db.Float, nullable=False),
        db.Column("y30", db.Float, nullable=False),
        db.Column("y31", db.Float, nullable=False),
        db.Column("y32", db.Float, nullable=False),
        db.Column("y33", db.Float, nullable=False),
        db.Column("y34", db.Float, nullable=False),
        db.Column("y35", db.Float, nullable=False),
        db.Column("y36", db.Float, nullable=False),
        db.Column("y37", db.Float, nullable=False),
        db.Column("y38", db.Float, nullable=False),
        db.Column("y39", db.Float, nullable=False),
        db.Column("y40", db.Float, nullable=False),
        db.Column("y41", db.Float, nullable=False),
        db.Column("y42", db.Float, nullable=False),
        db.Column("y43", db.Float, nullable=False),
        db.Column("y44", db.Float, nullable=False),
        db.Column("y45", db.Float, nullable=False),
        db.Column("y46", db.Float, nullable=False),
        db.Column("y47", db.Float, nullable=False),
        db.Column("y48", db.Float, nullable=False),
        db.Column("y49", db.Float, nullable=False),
        db.Column("y50", db.Float, nullable=False)
    )
    meta_data.create_all(engine)

    # Inserting into created table
    ideal_func = pd.read_csv(
        filepath_or_buffer="C:\\Users\\user\\PycharmProjects\\python_assignment\\datasets\\ideal.csv")
    df = pd.DataFrame(ideal_func)
    ideal_table = db.Table("ideal", meta_data, autoload=True, autoload_with=engine)
    for i, row in ideal_func.iterrows():
        sql_query = "INSERT INTO details.ideal VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s," \
                    "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        connection.execute(sql_query, row)
        print("Record inserted successfully")

#  print(best_func2)

if __name__ == '__main__':
    main()
    details_train()
    details_ideal()
