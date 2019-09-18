import math, numpy as np

def parse_input(input_line):
    input_line = input_line.strip().split(" ")
    attributes = []
    for i in range(len(input_line)-1):
        s = input_line[i]
        if s.endswith(":"):
            attributes.append(int(s[:-1]))
        else:
            attributes.append(int(s))
    y = int(input_line[-1])
    return attributes, y

def get_data_from_file(filename):
    #Get the data
    file = open(filename,"r")
    num_parameters = int(file.readline().strip())
    num_users = int(file.readline().strip())

    #Read data into matrix
    x_matrix = []
    y_vector = []
    for i in range(num_users):
        attributes, y = parse_input(file.readline())
        x_matrix.append(attributes)
        y_vector.append(y)
    return x_matrix, y_vector

def weighted_sum(theta, x):
    x = np.array(x)
    theta = np.array(theta)
    return np.dot(theta,x)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Choose parameters of gradient that maximize the likelihood
def logistic_regression(x_matrix, y_vector, step_size, num_iterations):
    num_parameters = len(x_matrix[0])
    num_data_points = len(x_matrix)
    theta = np.zeros(num_parameters)
    for i in range(num_iterations):
        gradient = np.zeros(num_parameters)
        for k in range(num_data_points):
            x = x_matrix[k]
            y = y_vector[k]
            for j in range(num_parameters):
                gradient[j] += x[j]*(y - sigmoid(weighted_sum(theta,x)))
        for j in range(num_parameters):
            theta[j] += step_size * gradient[j]
    return theta


def test_model(x_matrix, y_vector, theta):
    accurate_count = 0
    for i in range(len(x_matrix)):
        x = x_matrix[i]
        y = y_vector[i]
        y_hat = weighted_sum(theta,x)
        y_hat = sigmoid(y_hat)
        if (y_hat>=0.5):
            y_hat = 1
        else:
            y_hat = 0
        if (y_hat == y):
            accurate_count += 1
    print(str(accurate_count) + "/" + str(len(x_matrix)))

# Magnitude of step size we take
step_size = 0.01
# Number of iterations for the model
num_iterations = 1000

# Train the model
x_matrix, y_vector = get_data_from_file("netflix-train.txt")
theta = logistic_regression(x_matrix, y_vector, step_size, num_iterations)

# Test the model
x_matrix, y_vector = get_data_from_file("netflix-test.txt")
test_model(x_matrix, y_vector, theta)

