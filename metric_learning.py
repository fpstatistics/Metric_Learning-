import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import norm



###############local constant estimator
def N_W_estimator(y, X, A, i):
    X_j = np.delete(X - X[i,:],i,axis=0)
    A_T_A = np.dot(A.T,A)
    X_A_T_A_X = np.diag(np.dot(np.dot(X_j,A_T_A),X_j.T))

    K =  norm.pdf( np.sqrt(2 * X_A_T_A_X ) )
    molecular = np.sum( K * np.delete(y,i) )
    denominator = np.sum(K)
    y_i_hat = molecular/ denominator
    return y_i_hat

def myouter(x):
    return np.outer(x, x)
def gradient(y, X, A, i):
    y_i_hat = N_W_estimator(y, X, A, i)
    X_j = np.delete(X - X[i, :], i, axis=0)
    X_T_X = np.apply_along_axis(myouter, 1, X_j)
    A_T_A = np.dot(A.T, A)
    X_A_T_A_X = np.diag(np.dot(np.dot(X_j, A_T_A), X_j.T))
    K = norm.pdf(np.sqrt(2 * X_A_T_A_X))
    dy = y_i_hat - np.delete(y, i)
    other = np.apply_along_axis(np.multiply, 0, X_T_X, K * dy)
    partial_A = 4 * np.dot(A, (y_i_hat - y[i]) * np.sum(other, 0))
    return partial_A

###use the local constant estimator to define the squre loss fuction
def Loss(y, X, A):
    n = np.shape(X)[0]
    y_hat = [N_W_estimator(y, X, A, i) for i in range(n)]
    #     y_hat = np.array(y_hat)
    sh1 = [i for i in range(n) if y_hat[i] < 1e13]  ##舍去那些估计量为零的样本点
    y_hat = np.array(y_hat)
    loss = np.sum((y[sh1] - y_hat[sh1]) ** 2)
    return loss


def SGD_LR(data_x, data_y, d, maxepochs=500,epsilon=1e-4):
    n, p = np.shape(data_x)
    A_ = np.eye(max(p,d))
    if p >= d:
        A = A_[0:d,:]
    else:
        A = A_[:,0:p]
#     A = np.array([[1,0],[0,1],[1,1]]).T
#     A = np.array([[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]]).T
    alpha_0 = 0.05
    epochs_count = 0
    loss_list = []
    epochs_list = []
    while (epochs_count < maxepochs ):
        rand_i = np.random.randint(n)  # 随机取一个样本
        loss = Loss(data_y,data_x,A)
#         print(loss)
        grad = gradient(data_y,data_x,A,rand_i) #损失函数的梯度
        A_temp = A - alpha_0 * grad
        loss_new_temp = Loss(data_y,data_x,A_temp)
        if( loss_new_temp - loss < 0 ): #要求迭代之后的损失函数必须下降，否则就重新随机一个样本在进行迭代
            A = A_temp
            loss_new = loss_new_temp
        else:
            loss_new = 1e10
        if abs(loss_new - loss) < epsilon:
            break
        epochs_list.append(epochs_count)
        epochs_count += 1
    loss = loss_new_temp

    return [A,loss,epochs_count]


def cv_dimension(K, X, Y):
    n, p = X.shape
    kf = KFold(n_splits=K, shuffle=False, random_state=12)
    A_list = []
    mean_error_list = []
    param_grid = [2, 3]
    i = 0
    for param in param_grid:
        error_list = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index, :], X[test_index, :]
            Y_train, Y_test = Y[train_index], Y[test_index]
            re = SGD_LR(X_train, Y_train, d=param, maxepochs=50, epsilon=1e-3)
            A = re[0]
            A_A_T = np.dot(A, A.T)
            A_A_T_inv = np.linalg.inv(A_A_T)
            A_P = np.dot(np.dot(A.T, A_A_T_inv), A)
            ###test_error

            y_est = [N_W_estimator(Y_test, X_test, A, i) for i in range(len(Y_test))]
            err = sum((Y_test - y_est) ** 2)
            error_list.append(err)
        mean_error_list.append(np.mean(error_list))
        i += 1
        print(str(i) + 'parameter has been searched')
    indx = mean_error_list.index(min(mean_error_list))
    dimension_hat = param_grid[indx]

    return [dimension_hat, mean_error_list]

if __name__ == "__main__":

    def g(x):
        return x[1] * (x[0] + x[1] + 1) + x[2] ** 2


    class Model:
        def __init__(self, N):
            p = 3
            self.p = p
            X = np.random.multivariate_normal(np.zeros(p), np.identity(p), N)
            self.X = X
            sigma_eps = 1 # 同方差情况
            self.eps = sigma_eps * np.random.normal(0, 1, N)
            self.g_X = g(self.X.T)
            Y = self.g_X + self.eps
            self.Y = Y

            L = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 0]])
            L_T_L = np.dot(L.T, L)
            L_T_L_inv = np.linalg.inv(L_T_L)
            L_P = np.dot(np.dot(L, L_T_L_inv), L.T)
            self.L_P = L_P

    L = np.array([[1, 0, 0], [1, 0, 1], [0, 1, 0]])
    L_T_L = np.dot(L.T, L)
    L_T_L_inv = np.linalg.inv(L_T_L)
    L_P = np.dot(np.dot(L, L_T_L_inv), L.T)
    r1 = []
    r2 = []
    def simulation(ntimes):
        for i in range(ntimes):
            model = Model(300)
            data_x = model.X
            data_y = model.Y
            d = cv_dimension(5, data_x, data_y)[0]
            r1.append(d)
            result = SGD_LR(data_x, data_y, d, maxepochs=50, epsilon=1e-3)
            A = result[0]
            if A.shape[0] != 3:
                r2.append('Null')
            else:
                A_A_T = np.dot(A, A.T)
                A_A_T_inv = np.linalg.inv(A_A_T)
                A_P = np.dot(np.dot(A.T, A_A_T_inv), A)
                dis = A_P - L_P
                r2.append(np.sqrt(np.trace(np.dot(dis.T, dis))))

            return [r1, r2]
    result = simulation(100)





