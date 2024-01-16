import torch
from numpy.random import choice


batches_per_epoch = 1
batch_size = 400
mse_loss = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()


class Net(torch.nn.Module):
    """Multi-Layer Perceptron for regression task
    """
    def __init__(self):
        """Class constructor
        """
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(in_features=13, out_features=64)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.25)
        self.linear2 = torch.nn.Linear(in_features=64, out_features=64)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(p=0.25)
        self.linear3 = torch.nn.Linear(in_features=64, out_features=1)

    def forward(self, inp):
        """Forward pass

        Args:
            inp (torch.tensor): input of the network

        Returns:
            torch.tensor: output of the network
        """
        x = self.linear1(inp)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        return (x)

    def training_SGD(
        self, X_train, y_train, X_test, y_test,
            n_epochs, alpha, verbose=False):
        """SGD optimization algorithm

        Args:
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float): learning rate
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train MSE, test MSE,
            train MAE, test MAE
        """
        if verbose:
            print('Launching SGD training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        mse_train_epochs = []
        mse_test_epochs = []
        mae_train_epochs = []
        mae_test_epochs = []
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)
                hat_yb = hat_yb.reshape(hat_yb.shape[0],)  # Forward pass:
                loss = mse_loss(hat_yb, yb)  # Forward pass : Compute the loss
                self.zero_grad()  # re-init the gradients
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for param in self.parameters():  # Parameters' update
                        param -= alpha * param.grad
                    mae_train = mae_loss(hat_yb, yb)
            mse_train_epochs.append(loss.detach().numpy())
            mae_train_epochs.append(mae_train)
            with torch.no_grad():
                hat_yt = self.forward(X_test)
                hat_yt = hat_yt.reshape(hat_yt.shape[0],)
                mse_test = mse_loss(hat_yt, y_test)
                mae_test = mae_loss(hat_yt, y_test)
                mse_test_epochs.append(mse_test)
                mae_test_epochs.append(mae_test)
            if verbose and T % 10 == 0:
                print(("".join([
                    'Epoch {} / {} : Loss train = {}',
                    '| MAE train = {} | Loss test = {}',
                    '| MAE test = {}'])).format(
                        T+1, n_epochs, loss, mae_train, mse_test, mae_test))
        return (
            mse_train_epochs, mse_test_epochs,
            mae_train_epochs, mae_test_epochs)

    def training_Mom(
        self, X_train, y_train, X_test, y_test,
            n_epochs, alpha, beta, verbose=False):
        """Momentum optimization algorithm

        Args:
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float): learning rate
            beta (float): weightage that is going to assign to the past values
            of the gradient
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train MSE, test MSE,
            train MAE, test MAE
        """
        mse_train_epochs = []
        mse_test_epochs = []
        mae_train_epochs = []
        mae_test_epochs = []
        if verbose:
            print('Launching Momentum training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        Momentums = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                hat_yb = hat_yb.reshape(hat_yb.shape[0],)
                loss = mse_loss(hat_yb, yb)  # Forward pass : Compute the loss
                self.zero_grad()  # re-init the gradients
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        Momentums[i] = beta*Momentums[i] + (1-beta)*param.grad
                        param -= alpha * Momentums[i]
                    mae_train = mae_loss(hat_yb, yb)
            mse_train_epochs.append(loss.detach().numpy())
            mae_train_epochs.append(mae_train)
            with torch.no_grad():
                hat_yt = self.forward(X_test)
                hat_yt = hat_yt.reshape(hat_yt.shape[0],)
                mse_test = mse_loss(hat_yt, y_test)
                mae_test = mae_loss(hat_yt, y_test)
                mse_test_epochs.append(mse_test)
                mae_test_epochs.append(mae_test)
            if verbose and T % 10 == 0:
                print(("".join([
                    'Epoch {} / {} : Loss train = {}',
                    '| MAE train = {} | Loss test = {}',
                    '| MAE test = {}'])).format(
                        T+1, n_epochs, loss, mae_train, mse_test, mae_test))
        return (
            mse_train_epochs, mse_test_epochs,
            mae_train_epochs, mae_test_epochs)

    def training_NAG(
        self, X_train, y_train, X_test, y_test,
            n_epochs, alpha, beta, verbose=False):
        """NAG optimization algorithm

        Args:
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float): learning rate
            beta (float): weightage that is going to assign to the past values
            of the gradient
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train MSE, test MSE,
            train MAE, test MAE
        """
        mse_train_epochs = []
        mse_test_epochs = []
        mae_train_epochs = []
        mae_test_epochs = []
        if verbose:
            print('Launching NAG training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        Momentums = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        param -= alpha*beta*Momentums[i]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                hat_yb = hat_yb.reshape(hat_yb.shape[0],)
                loss = mse_loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        Momentums[i] = beta*Momentums[i] + (1-beta)*param.grad
                        param -= alpha*(1-beta)*param.grad
                    mae_train = mae_loss(hat_yb, yb)
            mse_train_epochs.append(loss.detach().numpy())
            mae_train_epochs.append(mae_train)
            with torch.no_grad():
                hat_yt = self.forward(X_test)
                hat_yt = hat_yt.reshape(hat_yt.shape[0],)
                mse_test = mse_loss(hat_yt, y_test)
                mae_test = mae_loss(hat_yt, y_test)
                mse_test_epochs.append(mse_test)
                mae_test_epochs.append(mae_test)
            if verbose and T % 10 == 0:
                print(("".join([
                    'Epoch {} / {} : Loss train = {}',
                    '| MAE train = {} | Loss test = {}',
                    '| MAE test = {}'])).format(
                        T+1, n_epochs, loss, mae_train, mse_test, mae_test))
        return (
            mse_train_epochs, mse_test_epochs,
            mae_train_epochs, mae_test_epochs)

    def training_ADG(
        self, X_train, y_train, X_test, y_test,
            n_epochs, alpha=0.01, epsilon=1e-8, verbose=False):
        """ADG optimization algorithm

        Args:
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float, optional): learning rate. Defaults to 0.01
            epsilon (float, optional): stabilization parameter. Defaults to
            1e-8
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train MSE, test MSE,
            train MAE, test MAE
        """
        mse_train_epochs = []
        mse_test_epochs = []
        mae_train_epochs = []
        mae_test_epochs = []
        if verbose:
            print('Launching AdaGrad training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        S = [torch.ones(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                hat_yb = hat_yb.reshape(hat_yb.shape[0],)
                loss = mse_loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        S[i] += param.grad.pow(2)
                        param -= torch.mul(
                            alpha/(epsilon + torch.sqrt(S[i])), param.grad)
                    mae_train = mae_loss(hat_yb, yb)
            mse_train_epochs.append(loss.detach().numpy())
            mae_train_epochs.append(mae_train)
            with torch.no_grad():
                hat_yt = self.forward(X_test)
                hat_yt = hat_yt.reshape(hat_yt.shape[0],)
                mse_test = mse_loss(hat_yt, y_test)
                mae_test = mae_loss(hat_yt, y_test)
                mse_test_epochs.append(mse_test)
                mae_test_epochs.append(mae_test)
            if verbose and T % 10 == 0:
                print(("".join([
                    'Epoch {} / {} : Loss train = {}',
                    '| MAE train = {} | Loss test = {}',
                    '| MAE test = {}'])).format(
                        T+1, n_epochs, loss, mae_train, mse_test, mae_test))
        return (
            mse_train_epochs, mse_test_epochs,
            mae_train_epochs, mae_test_epochs)

    def training_RMS(
        self, X_train, y_train, X_test, y_test, n_epochs,
            alpha=0.01, gamma=0.9, epsilon=1e-8, verbose=False):
        """RMS Prop optimization algorithm

        Args:
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float, optional): learning rate. Defaults to 0.01
            gamma (float, optional): decay factor. Defaults to 0.9
            epsilon (float, optional): stabilization parameter. Defaults to
            1e-8
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train MSE, test MSE,
            train MAE, test MAE
        """
        mse_train_epochs = []
        mse_test_epochs = []
        mae_train_epochs = []
        mae_test_epochs = []
        if verbose:
            print('Launching RMS Prop training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        S = [torch.ones(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                hat_yb = hat_yb.reshape(hat_yb.shape[0],)
                loss = mse_loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        S[i] = gamma * S[i] + (1-gamma) * param.grad.pow(2)
                        param -= torch.mul(
                            alpha/(epsilon + torch.sqrt(S[i])), param.grad)
                    mae_train = mae_loss(hat_yb, yb)
            mse_train_epochs.append(loss.detach().numpy())
            mae_train_epochs.append(mae_train)
            with torch.no_grad():
                hat_yt = self.forward(X_test)
                hat_yt = hat_yt.reshape(hat_yt.shape[0],)
                mse_test = mse_loss(hat_yt, y_test)
                mae_test = mae_loss(hat_yt, y_test)
                mse_test_epochs.append(mse_test)
                mae_test_epochs.append(mae_test)
            if verbose and T % 10 == 0:
                print(("".join([
                    'Epoch {} / {} : Loss train = {}',
                    '| MAE train = {} | Loss test = {}',
                    '| MAE test = {}'])).format(
                        T+1, n_epochs, loss, mae_train, mse_test, mae_test))
        return (
            mse_train_epochs, mse_test_epochs,
            mae_train_epochs, mae_test_epochs)

    def training_ADD(
        self, X_train, y_train, X_test, y_test,
            n_epochs, rho, epsilon=1e-8, verbose=False):
        """ADD optimization algorithm

        Args:
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            rho (float): decay factor
            epsilon (float, optional): stabilization parameter. Defaults to
            1e-8
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train MSE, test MSE,
            train MAE, test MAE
        """
        mse_train_epochs = []
        mse_test_epochs = []
        mae_train_epochs = []
        mae_test_epochs = []
        if verbose:
            print('Launching AdaDelta training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        S = [torch.zeros(param.shape) for param in self.parameters()]
        U = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)
                hat_yb = hat_yb.reshape(hat_yb.shape[0],)
                loss = mse_loss(hat_yb, yb)
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        S[i] = rho * S[i] + (1 - rho) * param.grad.pow(2)
                        D_param = - torch.mul(
                            (epsilon + torch.sqrt(U[i]))/(
                                epsilon + torch.sqrt(S[i])), param.grad)
                        U[i] = rho * U[i] + (1 - rho) * D_param.pow(2)
                        param += D_param
                    mae_train = mae_loss(hat_yb, yb)
            mse_train_epochs.append(loss.detach().numpy())
            mae_train_epochs.append(mae_train)
            with torch.no_grad():
                hat_yt = self.forward(X_test)
                hat_yt = hat_yt.reshape(hat_yt.shape[0],)
                mse_test = mse_loss(hat_yt, y_test)
                mae_test = mae_loss(hat_yt, y_test)
                mse_test_epochs.append(mse_test)
                mae_test_epochs.append(mae_test)
            if verbose and T % 10 == 0:
                print(("".join([
                    'Epoch {} / {} : Loss train = {}',
                    '| MAE train = {} | Loss test = {}',
                    '| MAE test = {}'])).format(
                        T+1, n_epochs, loss, mae_train, mse_test, mae_test))
        return (
            mse_train_epochs, mse_test_epochs,
            mae_train_epochs, mae_test_epochs)

    def training_ADAM(
        self, X_train, y_train, X_test, y_test,
            n_epochs, alpha, beta1, beta2, epsilon=1e-8, verbose=False):
        """Adam optimization algorithm

        Args:
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float, optional): learning rate. Defaults to 0.01
            beta1 (float): exponential decay rate for the first moment estimate
            beta2 (float): exponential decay rate for the second moment
            estimate
            epsilon (float, optional): stabilization parameter. Defaults to
            1e-8
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train MSE, test MSE,
            train MAE, test MAE
        """
        mse_train_epochs = []
        mse_test_epochs = []
        mae_train_epochs = []
        mae_test_epochs = []
        if verbose:
            print('Launching Adam training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        V = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                hat_yb = hat_yb.reshape(hat_yb.shape[0],)
                loss = mse_loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        M[i] = beta1 * M[i] + (1-beta1) * param.grad
                        V[i] = beta2 * V[i] + (1-beta2) * param.grad.pow(2)
                        M_hat = (1/(1-beta1**(T+1))) * M[i]
                        V_hat = (1/(1-beta2**(T+1))) * V[i]
                        param -= alpha / (torch.sqrt(V_hat) + epsilon) * M_hat
                    mae_train = mae_loss(hat_yb, yb)
            mse_train_epochs.append(loss.detach().numpy())
            mae_train_epochs.append(mae_train)
            with torch.no_grad():
                hat_yt = self.forward(X_test)
                hat_yt = hat_yt.reshape(hat_yt.shape[0],)
                mse_test = mse_loss(hat_yt, y_test)
                mae_test = mae_loss(hat_yt, y_test)
                mse_test_epochs.append(mse_test)
                mae_test_epochs.append(mae_test)
            if verbose and T % 10 == 0:
                print(("".join([
                    'Epoch {} / {} : Loss train = {}',
                    '| MAE train = {} | Loss test = {}',
                    '| MAE test = {}'])).format(
                        T+1, n_epochs, loss, mae_train, mse_test, mae_test))
        return (
            mse_train_epochs, mse_test_epochs,
            mae_train_epochs, mae_test_epochs)

    def training_AMS(
        self, X_train, y_train, X_test, y_test,
            n_epochs, alpha, beta1, beta2, epsilon=1e-8, verbose=False):
        """AMS optimization algorithm

        Args:
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float, optional): learning rate. Defaults to 0.01
            beta1 (float): exponential decay rate for the first moment estimate
            beta2 (float): exponential decay rate for the second
            moment estimate
            epsilon (float, optional): stabilization parameter. Defaults to
            1e-8
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train MSE, test MSE,
            train MAE, test MAE
        """
        mse_train_epochs = []
        mse_test_epochs = []
        mae_train_epochs = []
        mae_test_epochs = []
        if verbose:
            print('Launching AMS Grad training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        V = [torch.zeros(param.shape) for param in self.parameters()]
        V_hat = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                hat_yb = hat_yb.reshape(hat_yb.shape[0],)
                loss = mse_loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        M[i] = beta1 * M[i] + (1-beta1) * param.grad
                        V[i] = beta2 * V[i] + (1-beta2) * param.grad.pow(2)
                        V_hat[i] = torch.maximum(V_hat[i], V[i])
                        param -= alpha / (torch.sqrt(V_hat[i]) + epsilon)*M[i]
                    mae_train = mae_loss(hat_yb, yb)
            mse_train_epochs.append(loss.detach().numpy())
            mae_train_epochs.append(mae_train)
            with torch.no_grad():
                hat_yt = self.forward(X_test)
                hat_yt = hat_yt.reshape(hat_yt.shape[0],)
                mse_test = mse_loss(hat_yt, y_test)
                mae_test = mae_loss(hat_yt, y_test)
                mse_test_epochs.append(mse_test)
                mae_test_epochs.append(mae_test)
            if verbose and T % 10 == 0:
                print(("".join([
                    'Epoch {} / {} : Loss train = {}',
                    '| MAE train = {} | Loss test = {}',
                    '| MAE test = {}'])).format(
                        T+1, n_epochs, loss, mae_train, mse_test, mae_test))
        return (
            mse_train_epochs, mse_test_epochs,
            mae_train_epochs, mae_test_epochs)

    def training_NADAM(
        self, X_train, y_train, X_test, y_test,
            n_epochs, alpha, mu, nu, epsilon=1e-8, verbose=False):
        """NADAM optimization algorithm

        Args:
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float, optional): learning rate. Defaults to 0.01
            mu (float): exponential decay rate for the first moment estimate
            nu (float): exponential decay rate for the second moment estimate
            epsilon (float, optional): stabilization parameter. Defaults to
            1e-8
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train MSE, test MSE,
            train MAE, test MAE
        """
        mse_train_epochs = []
        mse_test_epochs = []
        mae_train_epochs = []
        mae_test_epochs = []
        if verbose:
            print('Launching Nadam training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        N = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                hat_yb = hat_yb.reshape(hat_yb.shape[0],)
                loss = mse_loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        M[i] = mu * M[i] + (1-mu) * param.grad
                        g_hat = 1/(1-mu**(T+1)) * param.grad
                        m_hat = (1/(1-mu**(T+1))) * M[i]
                        N[i] = nu * N[i] + (1-nu) * param.grad.pow(2)
                        n_hat = 1/(1-nu**(T+1)) * N[i]
                        m_bar = mu * m_hat + (1-mu) * g_hat
                        param -= alpha / (torch.sqrt(n_hat) + epsilon) * m_bar
                    mae_train = mae_loss(hat_yb, yb)
            mse_train_epochs.append(loss.detach().numpy())
            mae_train_epochs.append(mae_train)
            with torch.no_grad():
                hat_yt = self.forward(X_test)
                hat_yt = hat_yt.reshape(hat_yt.shape[0],)
                mse_test = mse_loss(hat_yt, y_test)
                mae_test = mae_loss(hat_yt, y_test)
                mse_test_epochs.append(mse_test)
                mae_test_epochs.append(mae_test)
            if verbose and T % 10 == 0:
                print(("".join([
                    'Epoch {} / {} : Loss train = {}',
                    '| MAE train = {} | Loss test = {}',
                    '| MAE test = {}'])).format(
                        T+1, n_epochs, loss, mae_train, mse_test, mae_test))
        return (
            mse_train_epochs, mse_test_epochs,
            mae_train_epochs, mae_test_epochs)

    def training_ADAMAX(
        self, X_train, y_train, X_test, y_test,
            n_epochs, alpha, beta1, beta2, lambd, epsilon=1e-8, verbose=False):
        """ADAMAX optimization algorithm

        Args:
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float, optional): learning rate. Defaults to 0.01
            beta1 (float): exponential decay rate for the first moment estimate
            beta2 (float): exponential decay rate for the second
            moment estimate
            lambd (float): regularization parameter
            epsilon (float, optional): stabilization parameter. Defaults to
            1e-8
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train MSE, test MSE,
            train MAE, test MAE
        """
        mse_train_epochs = []
        mse_test_epochs = []
        mae_train_epochs = []
        mae_test_epochs = []
        if verbose:
            print('Launching Adamax training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        U = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)
                hat_yb = hat_yb.reshape(hat_yb.shape[0],)
                loss = mse_loss(hat_yb, yb)
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        g = param.grad
                        if lambd != 0:
                            g = g + lambd * param
                        M[i] = beta1 * M[i] + (1-beta1) * param.grad
                        U[i] = torch.maximum(beta2*U[i], torch.abs(g)+epsilon)
                        param -= alpha / ((1-beta1**(T+1))*U[i]) * M[i]
                    mae_train = mae_loss(hat_yb, yb)
            mse_train_epochs.append(loss.detach().numpy())
            mae_train_epochs.append(mae_train)
            with torch.no_grad():
                hat_yt = self.forward(X_test)
                hat_yt = hat_yt.reshape(hat_yt.shape[0],)
                mse_test = mse_loss(hat_yt, y_test)
                mae_test = mae_loss(hat_yt, y_test)
                mse_test_epochs.append(mse_test)
                mae_test_epochs.append(mae_test)
            if verbose and T % 10 == 0:
                print(("".join([
                    'Epoch {} / {} : Loss train = {}',
                    '| MAE train = {} | Loss test = {}',
                    '| MAE test = {}'])).format(
                        T+1, n_epochs, loss, mae_train, mse_test, mae_test))
        return (
            mse_train_epochs, mse_test_epochs,
            mae_train_epochs, mae_test_epochs)

    def training_NOS(
        self, X_train, y_train, X_test, y_test,
            n_epochs, alpha,
            beta1, gamma, epsilon=1e-8, verbose=False):
        """Nostalgic Adam optimization algorithm

        Args:
            Loss (torch.nn): loss function
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            batch_size (int): batch size
            batches_per_epoch (int): number of batch per epoch
            alpha (float): learning rate
            beta1 (float): exponential decay rate for the first moment estimate
            gamma (float): decay factor
            epsilon (float, optional): stabilization parameter. Defaults to
            1e-8
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy,
            test loss, test accuracy
        """
        mse_train_epochs = []
        mse_test_epochs = []
        mae_train_epochs = []
        mae_test_epochs = []
        if verbose:
            print('Launching Nostalgic Adam training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        V = [torch.zeros(param.shape) for param in self.parameters()]
        B1, B2 = 0, 1
        for T in range(1, n_epochs+1):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                hat_yb = hat_yb.reshape(hat_yb.shape[0],)
                loss = mse_loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        beta2 = B1/B2
                        B1 += T**(-gamma)
                        B2 += (T+1)**(-gamma)
                        M[i] = beta1 * M[i] + (1-beta1) * param.grad
                        V[i] = beta2 * V[i] + (1-beta2) * param.grad.pow(2)
                        M_hat = (1/(1-beta1**(T+1))) * M[i]
                        V_hat = (1/(1-beta2**(T+1))) * V[i]
                        param -= alpha / (torch.sqrt(V_hat) + epsilon) * M_hat
                    mae_train = mae_loss(hat_yb, yb)
            mse_train_epochs.append(loss.detach().numpy())
            mae_train_epochs.append(mae_train)
            with torch.no_grad():
                hat_yt = self.forward(X_test)
                hat_yt = hat_yt.reshape(hat_yt.shape[0],)
                mse_test = mse_loss(hat_yt, y_test)
                mae_test = mae_loss(hat_yt, y_test)
                mse_test_epochs.append(mse_test)
                mae_test_epochs.append(mae_test)
            if verbose and T % 10 == 0:
                print(("".join([
                    'Epoch {} / {} : Loss train = {}',
                    '| MAE train = {} | Loss test = {}',
                    '| MAE test = {}'])).format(
                        T+1, n_epochs, loss, mae_train, mse_test, mae_test))
        return (
            mse_train_epochs, mse_test_epochs,
            mae_train_epochs, mae_test_epochs)
