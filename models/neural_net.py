"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C.
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last.
    The outputs of the last fully-connected layer are passed through
    a sigmoid.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
            opt: option for using "SGD" or "Adam" optimizer (Adam is Extra Credit)
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers
        self.opt = opt

        # layers are like [hidden layer 1, ..., hidden layer k, output layer]
        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1],
                                                        sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

            # TODO: (Extra Credit) You may set parameters for Adam optimizer here

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """

        # just need to do W.X + B.
        # W is (D, H), X is (N, D), b is (H,)
        # print(f"In Linear: X: {X.shape} W: {W.shape} B {b.shape}")

        out = X @ W + b
        return out

    def linear_grad(self, W: np.ndarray, X: np.ndarray, de_dz: np.ndarray) -> np.ndarray:
        """Gradient of linear layer
        Parameters:
            W: the weight matrix
            X: the input data
            de_dz: the gradient of loss
        Returns:
            de_dw, de_db, de_dx
            where
                de_dw: gradient of loss with respect to W
                de_db: gradient of loss with respect to b
                de_dx: gradient of loss with respect to X
        """
        # print(f"Linear grad, W {W.shape} X: {X.shape} de_dz: {de_dz.shape}")

        # TODO do i really understand these?
        de_dw = X.T @ de_dz
        de_dx = de_dz @ W.T
        de_db = np.sum(de_dz, axis=0)

        # print(
        # f"Linear grad, de_dw {de_dw.shape} de_dx: {de_dx.shape} de_db: {de_db.shape}")

        return de_dw, de_db, de_dx

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """

        return np.maximum(X, 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        grad = np.where(X > 0, 1, 0)

        return grad

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # print(f"In sigmoid, x: {x.shape}")

        # pos_mask = (x >= 0)
        # neg_mask = (x < 0)

        # denominator = np.zeros_like(x)
        # denominator[pos_mask] = np.exp(-x[pos_mask])
        # denominator[neg_mask] = np.exp(x[neg_mask])
        # numerator = np.ones_like(x)
        # numerator[neg_mask] = denominator[neg_mask]

        # return numerator / (1 + denominator)

        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        # print(f"In sigmoid grad , X: {X.shape}")
        sig = self.sigmoid(X)
        grad = sig * (1 - sig)
        # print(f"Gradient: {grad.shape}")
        return grad

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.mean((y - p) ** 2)

    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (2 / y.shape[0]) * (p - y)

    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        mse_grad = self.mse_grad(y, p)
        sig_grad = self.sigmoid_grad(p)
        return mse_grad * sig_grad

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C)
        """
        self.outputs = {"X0": X}
        for i in range(1, self.num_layers):
            X = self.relu(self.linear(
                self.params[f"W{i}"], X, self.params[f"b{i}"]))
            self.outputs[f"X{i}"] = X
        X = self.sigmoid(self.linear(
            self.params[f"W{self.num_layers}"], X, self.params[f"b{self.num_layers}"]))
        self.outputs[f"X{self.num_layers}"] = X
        return X

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        loss = self.mse(y, self.outputs[f"X{self.num_layers}"])
        de_dz = self.mse_sigmoid_grad(y, self.outputs[f"X{self.num_layers}"])

        for i in range(self.num_layers, 0, -1):
            W, X = self.params[f"W{i}"], self.outputs[f"X{i-1}"]
            de_dw, de_db, de_dz = self.linear_grad(W, X, de_dz)
            self.gradients[f"W{i}"] = de_dw
            self.gradients[f"b{i}"] = de_db
            if i > 1:
                de_dz *= self.relu_grad(self.outputs[f"X{i-1}"])

        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter(for Adam)
            b2: beta 2 parameter(for Adam)
            eps: epsilon to prevent division by zero(for Adam)
        """
        if self.opt == 'SGD':
            for i in range(1, self.num_layers+1):
                self.params[f"W{i}"] -= lr * self.gradients[f"W{i}"]
                self.params[f"b{i}"] -= lr * self.gradients[f"b{i}"]
        elif self.opt == 'Adam':
            # TODO: (Extra credit) implement Adam optimizer here
            pass
        else:
            raise NotImplementedError
