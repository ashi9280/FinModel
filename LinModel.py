import numpy as np

def least_squares(p):
    # p is a list of points x1,... , xn, y
    y = p[:, -1]
    X = p[:, :-1]

    # add a column of ones to X
    X = np.column_stack((np.ones(X.shape[0]), X))


    try:
        coeff = fit_model(X, y)
        return coeff
    except:
        print("Error: Matrix is not invertible, try more points")
        return np.zeros(X.shape[1])


def autoregression_least_squares(y, p):
    # y is y1, y2, y3, ...., yn
    # p is the order of the model

    y_rev = y[::-1]

    # y_rev is yn, yn-1, yn-2, ...., y1

    # get the last p+1 elements of y_rev
    Y = np.lib.stride_tricks.sliding_window_view(y_rev, (p+1,))[::-1]

    y = Y[:, 0]
    A = Y[:, 1:][:, ::-1]
    A = np.column_stack((np.ones(A.shape[0]), A))

    try:
        coeff = fit_model(A, y)
        return coeff
    except:
        print("Error: Matrix is not invertible, try more points")
        return np.zeros(p+1)


def fit_model(X, y):

    A_T = X.T
    A_T_A = A_T @ X
    A_T_y = A_T @ y

    A_T_A_inv = np.linalg.inv(A_T_A)
    alpha = A_T_A_inv @ A_T_y

    return alpha

X = np.array([[1, 0], [1, 1], [1, 2]])
y = np.array([6, 0, 0])
# print(fit_model(X, y))



p = np.array([[-6, -1, 1], [-2, 2, 5], [1, 1, 7], [7, 6, 4]])
# print(least_squares(p))



y = np.array([1, 2, 2, 4, 5.1, 7, 4, 3, 5, 9, 13])
# print(autoregression_least_squares(y, 4))
