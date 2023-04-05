#!/usr/bin/env python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

from encoders import PCAEncoder
from kernels import GaussianRBFKernel, LinearKernel, PolynomialKernel
from linear_models import (
    LinearModel,
    LinearClassifier,
    KernelClassifier,
)
from optimizers import (
    GradientDescent,
    GradientDescentLineSearch,
    StochasticGradient,
)
from fun_obj import (
    LeastSquaresLoss,
    KernelLogisticRegressionLossL2,
)
from learning_rate_getters import (
    ConstantLR,
)
from utils import (
    load_dataset,
    load_trainval,
    load_and_split,
    plot_classifier,
    savefig,
    standardize_cols,
    handle,
    run,
    main,
)


@handle("1")
def q1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    # kernel logistic regression with a linear kernel
    loss_fn = KernelLogisticRegressionLossL2(1)
    optimizer = GradientDescentLineSearch()
    kernel = LinearKernel()
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    #fig = plot_classifier(klr_model, X_train, y_train)
    #savefig("logRegLinear.png", fig)

    Gc = np.array([[-12, 12],[12, -12]])
    # print(Gc)
    w1 = np.array([[1, 1]])
    #ans = np.linalg.solve((Gc - np.eye(2)*28 ),[[0 , 0]])
    #print('w1 = ', w1)
    w2 = np.array([[1],[3]])
    #print('w2 = ', w2)
    print('dot = ', np.dot(w1,w2))
    w1 = np.array([1, 1])
    w2 = np.array([0, 0])
    mu = np.array([4, 2])


   # Given data
    w1 = np.array([1, 1])
    w2 = np.array([-1, 1])
    mean_vector = np.array([4, 2])
    X = np.array([[2, 0], [3, 1], [7, 5], [3, 3], [5, 1]])
    Z_tilde = np.array([4, 2])
    x_test = np.array([5, 5])
    
    #print X
    
    # Compute reconstructions
    W = np.array([w1, w2])
    X_tilde = np.dot(Z_tilde, W) + mean_vector
    x_test_tilde = np.dot(x_test - mean_vector, W.T) + mean_vector

    error = np.linalg.norm(Z_tilde @ W - ( x_test - mean_vector))
    print('error = ', error)
    # Compute L2 reconstruction errors
    errors_X = np.linalg.norm(X - X_tilde, axis=1)
    error_x_test = np.linalg.norm(x_test - x_test_tilde)

    # Plot the data with errors
    plt.scatter(X[:, 0], X[:, 1], color='blue', label='Original data')
    plt.scatter(X_tilde[:, 0], X_tilde[:, 1], color='red', label='Reconstructed data')
    plt.scatter(x_test[0], x_test[1], color='green', label='Test point')
    plt.errorbar(X[:, 0], X[:, 1], xerr=errors_X, yerr=errors_X, fmt='none', ecolor='blue', alpha=0.5)
    plt.errorbar(X_tilde[:, 0], X_tilde[:, 1], xerr=errors_X, yerr=errors_X, fmt='none', ecolor='red', alpha=0.5)
    plt.errorbar(x_test[0], x_test[1], xerr=error_x_test, yerr=error_x_test, fmt='none', ecolor='green', alpha=0.5)
    plt.legend()
    plt.show()

    # # Define data
    # X = np.array([[2, 0], [3, 1], [7, 5], [3, 3], [5, 1]])
    # mu = np.mean(X, axis=0)
    # print('mu', mu)
    # x_tilde = np.array([5, 5])
    # w1 = np.array([1, 1])
    # w2 = np.array([-1, 1])

    # # Center data and test point
    # Xc = X - mu
    # print('Xc= ', Xc)
    # x_tilde1_c = x_tilde - mu
    # x_tilde1_c = x_tilde1_c[:, np.newaxis]
    # print('x tilde = ', x_tilde1_c)

    # # Compute projections onto principal components
    # W = np.vstack((w1, w2))
    # print('x tildec shape = ', x_tilde1_c.shape)
    # print('W shape = ', W.shape)
    # print('W = ', W)

    # z_tilde = W@(x_tilde1_c)
    # print('z_tilde = ', z_tilde)
    # z_tilde1_w1 = [z_tilde[0], 0] #z_tilde[0] @ w1.T #np.dot(x_tilde1_c, w1) / np.linalg.norm(w1)
    # z_tilde1_w2 = [0, z_tilde[1]]#z_tilde[1]# * w2
    # print('z_tildew1 = ', z_tilde1_w1)
    # print('z_tildew1 = ', z_tilde1_w2)
    # print('mu = ', mu)

    # # Compute reconstructions in original coordinate system
    # x_tilde_hat1 = mu + z_tilde1_w1 * w1
    # x_tilde_hat2 = mu + z_tilde1_w2 * w2
    # print('x_tilde_hat1 = ', x_tilde_hat1)
    # print('x_tilde_hat1 = ', x_tilde_hat2)

    # # Compute L2 reconstruction errors
    # L2_error1 = np.linalg.norm(x_tilde - x_tilde_hat1)
    # L2_error2 = np.linalg.norm(x_tilde - x_tilde_hat2)

    # # Plot PCA coordinate system
    # plt.plot([mu[0], mu[0]+w1[0]], [mu[1], mu[1]+w1[1]], '-g', label='w1')
    # plt.plot([mu[0], mu[0]+w2[0]], [mu[1], mu[1]+w2[1]], '-b', label='w2')

    # # Plot projections and reconstructions
    # plt.plot([x_tilde[0], x_tilde_hat1[0]], [x_tilde[1], x_tilde_hat1[1]], '-r', label='Projection onto w1')
    # plt.plot([x_tilde[0], x_tilde_hat2[0]], [x_tilde[1], x_tilde_hat2[1]], '-m', label='Projection onto w2')
    # plt.plot(x_tilde[0], x_tilde[1], 'ro', label='Test point')
    # plt.plot(x_tilde_hat1[0], x_tilde_hat1[1], 'go', label='Reconstruction with w1')
    # plt.plot(x_tilde_hat1[0], x_tilde_hat2[1], 'bo', label='Reconstruction with w2')

    # plt.legend()
    #plt.show()

    # # Print reconstructions and L2 reconstruction errors
    # print("Reconstruction with w1: ", x_tilde_hat1)
    # print("L2 reconstruction error with w1: ", L2_error1)
    # print("Reconstruction with w2: ", x_tilde_hat2)
    # print("L2 reconstruction error with w2: ", L2_error2)
@handle("1.1")
def q1_1():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    #TODO YOUR CODE HERE FOR Q1.1
    #raise NotImplementedError
    
    # For polynomial Kernal:
    p, sigma, lammy = 2, 0.5, 0.01
    loss_fn = KernelLogisticRegressionLossL2(lammy)
    optimizer = GradientDescentLineSearch()
    kernel = PolynomialKernel(p)
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)

    print('----- For Polynomial Kernel -----')
    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegPOLY.png", fig)

    print('----- For RBF Kernel -----')
    kernel = GaussianRBFKernel(sigma)
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)
    print(f"Training error {np.mean(klr_model.predict(X_train) != y_train):.1%}")
    print(f"Validation error {np.mean(klr_model.predict(X_val) != y_val):.1%}")

    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("logRegRBF.png", fig)




@handle("1.2")
def q1_2():
    X_train, y_train, X_val, y_val = load_and_split("nonLinearData.pkl")

    sigmas = 10.0 ** np.array([-2, -1, 0, 1, 2])
    lammys = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])

    # train_errs[i, j] should be the train error for sigmas[i], lammys[j]
    train_errs = np.full((len(sigmas), len(lammys)), 100.0)
    val_errs = np.full((len(sigmas), len(lammys)), 100.0)  # same for val

    #TODO YOUR CODE HERE FOR Q1.2
    #raise NotImplementedError

    optimizer = GradientDescentLineSearch()
    for s in range(len(sigmas)):
        kernel = GaussianRBFKernel(sigmas[s])
        for l in range(len(lammys)):
            loss_fn = KernelLogisticRegressionLossL2(lammys[l])
            klr_model = KernelClassifier(loss_fn, optimizer, kernel)
            klr_model.fit(X_train, y_train)
            train_errs[s][l] = np.mean(klr_model.predict(X_train) != y_train)
            val_errs[s][l] = np.mean(klr_model.predict(X_val) != y_val)

    print('---- For training error: ----')
    train_min = np.unravel_index(np.argmin(train_errs),train_errs.shape)
    print('with best error of = ', np.min(train_errs))
    min_sigma = sigmas[train_min[0]]
    print('best sigma = ', min_sigma)
    min_lambda = lammys[train_min[1]]
    print('best lambda = ', min_lambda)


    print('---- For Validation error: ----')
    val_min = np.unravel_index(np.argmin(val_errs),val_errs.shape)
    print('with best error of = ', np.min(val_errs))
    print('best sigma = ', sigmas[val_min[0]])
    print('best lambda = ', lammys[val_min[1]])

    # decision plot
    
    loss_fn = KernelLogisticRegressionLossL2(min_lambda)
    kernel = GaussianRBFKernel(min_sigma)
    klr_model = KernelClassifier(loss_fn, optimizer, kernel)
    klr_model.fit(X_train, y_train)
    
    fig = plot_classifier(klr_model, X_train, y_train)
    savefig("hypersearch.png", fig)


    # Make a picture with the two error arrays. No need to worry about details here.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))
    for (name, errs), ax in zip([("training", train_errs), ("val", val_errs)], axes):
        cax = ax.matshow(errs, norm=norm)

        ax.set_title(f"{name} errors")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([str(sigma) for sigma in sigmas])
        ax.set_xlabel(r"$\lambda$")
        ax.set_xticks(range(len(lammys)))
        ax.set_xticklabels([str(lammy) for lammy in lammys])
        ax.xaxis.set_ticks_position("bottom")
    fig.colorbar(cax)
    savefig("logRegRBF_grids.png", fig)


@handle("3.2")
def q3_2():
    data = load_dataset("animals.pkl")
    X_train = data["X"]
    animal_names = data["animals"]
    trait_names = data["traits"]

    # Standardize features
    X_train_standardized, mu, sigma = standardize_cols(X_train)
    n, d = X_train_standardized.shape
    #print(X_train_standardized.shape)

    # Matrix plot
    fig, ax = plt.subplots()
    ax.imshow(X_train_standardized)
    savefig("animals_matrix.png", fig)
    plt.close(fig)

    # 2D visualization
    np.random.seed(3164)  # make sure you keep this seed
    #furry = trait_names[furry]
    print('furry = ', trait_names)
    print('furry = ', trait_names[11])
    print('furry = ', trait_names[57])
    j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    random_is = np.random.choice(n, 15, replace=False)  # choose random examples
    #j1, j2 = 11, 57
    fig, ax = plt.subplots()
    ax.scatter(X_train_standardized[:, j1], X_train_standardized[:, j2])
    ax.set_xlabel(trait_names[j1])
    ax.set_ylabel(trait_names[j2])
    for i in random_is:
        xy = X_train_standardized[i, [j1, j2]]
        ax.annotate(animal_names[i], xy=xy)
    savefig("animals_random.png", fig)
    plt.close(fig)
    
    #TODO YOUR CODE HERE FOR Q3.2 AND Q3.3
    k = 2
    X_train_standardized, mu, sigma = standardize_cols(X_train)
    n, d = X_train_standardized.shape
    print("X shape = ", X_train.shape)
    print("Xc shape = ", X_train_standardized.shape)
    model = PCAEncoder(13)
    model.fit(X_train_standardized)
    print("W shape = ", model.W.shape)
    z_hat = model.encode(X_train_standardized) # Zhat
    print('Z values = \n', X_train_standardized)
    #z_hat @ model.W
    print("z_hat shape = ", z_hat.shape)
    x_hat = model.decode(z_hat)
    
    fig, ax = plt.subplots()
    ax.scatter(z_hat[:,0], z_hat[:,1])
    ax.set_xlabel(trait_names[j1])
    ax.set_ylabel(trait_names[j2])

    for i in random_is:
        xy = z_hat[i, [0,1]]
        ax.annotate(animal_names[i], xy=xy)
    savefig("latentAnimals.png", fig)
    plt.close(fig)

    pc_1_in = np.argmax(np.abs(model.W[0]))
    pc_2_in = np.argmax(np.abs(model.W[1]))
    pc_1 = trait_names[pc_1_in]
    pc_2 = trait_names[pc_2_in]

    print('pc_1 = ', pc_1)
    print('pc_2 = ', pc_2)

    var = np.square(np.linalg.norm(z_hat @ model.W - X_train_standardized)) / np.square(np.linalg.norm((X_train_standardized)))
    print('variance = ', var)
    print('variance explained = ', 1-var)
    
    #plt.scatter(x_hat[:,0], x_hat[:,1])

    # # 2D visualization
    # np.random.seed(3164)  # make sure you keep this seed
    # j1, j2 = np.random.choice(d, 2, replace=False)  # choose 2 random features
    # random_is = np.random.choice(n, 15, replace=False)  # choose random examples

    # print('trait x = ', trait_names[j1])
    # print('trait y = ', trait_names[j2])
    # fig, ax = plt.subplots()
    # ax.scatter(z_hat[:, j1], z_hat[:, j2])
    # a, b = np.polyfit(z_hat[:, j1], z_hat[:, j2], 1)
    # plt.plot(z_hat[:, j1], a*z_hat[:, j1]+b)
    
    # for i in random_is:
    #     xy = z_hat[i, [j1, j2]]
    #     ax.annotate(animal_names[i], xy=xy)
    # savefig("animals_random_encoded.png", fig)
    # plt.close(fig)
    # #print(z_hat)
    


if __name__ == "__main__":
    main()
