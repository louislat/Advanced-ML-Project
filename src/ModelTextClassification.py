import torch
from numpy.random import choice


class Net(torch.nn.Module):
    """Multi-Layer Perceptron for text classification
    """
    def __init__(self):
        """Class constructor
        """
        super(Net, self).__init__()

        self.mlp1 = torch.nn.Linear(in_features=10000, out_features=16)
        self.relu1 = torch.nn.ReLU()
        self.mlp2 = torch.nn.Linear(in_features=16, out_features=16)
        self.relu2 = torch.nn.ReLU()
        self.mlp3 = torch.nn.Linear(in_features=16, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inp):
        """Forward pass

        Args:
            inp (torch.tensor): input of the network

        Returns:
            torch.tensor: output of the network
        """
        x = self.mlp1(inp)
        x = self.relu1(x)
        x = self.mlp2(x)
        x = self.relu2(x)
        x = self.mlp3(x)
        x = self.sigmoid(x)
        return (x)

    def training_SGD(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha, verbose=False):
        """SGD optimization algorithm

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
            verbose (bool, optional): parameter which prints messages on the standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy, test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching SGD training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, yb)  # Forward pass : Compute the loss
                self.zero_grad()  # re-init the gradients
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for param in self.parameters():  # Parameters' update
                        param -= alpha * param.grad
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = torch.mean(((hat_y > 0.5).float() == y_train).float())
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = torch.mean(
                    ((output_test > 0.5).float() == y_test).float())
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, loss))
        if verbose:
            print('Loss test = {}, Accuracy test = {}'.format(lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_Mom(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha, beta, verbose=False):
        """Momentum optimization algorithm

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
            beta (float): weightage that is going to assign to the past values of the gradient
            verbose (bool, optional): parameter which prints messages on the standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy, test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching Momentum training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        Momentums = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, yb)  # Forward pass : Compute the loss
                self.zero_grad()  # re-init the gradients
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        Momentums[i] = beta*Momentums[i] + (1-beta)*param.grad
                        param -= alpha * Momentums[i]
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = torch.mean(((hat_y > 0.5).float() == y_train).float())
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = torch.mean(
                    ((output_test > 0.5).float() == y_test).float())
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, loss))
        if verbose:
            print('Loss test = {}, Accuracy test = {}'.format(lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_NAG(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha, beta, verbose=False):
        """NAG optimization algorithm

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
            beta (float): weightage that is going to assign to the past values of the gradient
            verbose (bool, optional): parameter which prints messages on the standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy, test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching NAG training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        Momentums = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()  # re-init the gradients
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        param -= alpha*beta*Momentums[i]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        Momentums[i] = beta*Momentums[i] + (1-beta)*param.grad
                        param -= alpha*(1-beta)*param.grad
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = torch.mean(((hat_y > 0.5).float() == y_train).float())
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = torch.mean(
                    ((output_test > 0.5).float() == y_test).float())
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, loss))
        if verbose:
            print('Loss test = {}, Accuracy test = {}'.format(lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_ADG(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha=0.01,
            epsilon=1e-8, verbose=False):
        """ADG optimization algorithm

        Args:
            Loss (torch.nn): loss function
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            batch_size (int): batch size
            batches_per_epoch (int): number of batch per epoch
            alpha (float, optional): learning rate. Defaults to 0.01
            epsilon (float, optional): stabilization parameter. Defaults to 1e-8
            verbose (bool, optional): parameter which prints messages on the standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy, test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching Ada Grad training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        S = [torch.ones(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()  # re-init the gradients
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        S[i] += param.grad.pow(2)
                        param -= torch.mul(
                            alpha/(epsilon + torch.sqrt(S[i])), param.grad)
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = torch.mean(((hat_y > 0.5).float() == y_train).float())
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = torch.mean(
                    ((output_test > 0.5).float() == y_test).float())
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, loss))
        if verbose:
            print('Loss test = {}, Accuracy test = {}'.format(lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_RMS(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha=0.01, gamma=0.9,
            epsilon=1e-8, verbose=False):
        """RMS Prop optimization algorithm

        Args:
            Loss (torch.nn): loss function
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            batch_size (int): batch size
            batches_per_epoch (int): number of batch per epoch
            alpha (float, optional): learning rate. Defaults to 0.01
            gamma (float, optional): decay factor. Defaults to 0.9
            epsilon (float, optional): stabilization parameter. Defaults to 1e-8
            verbose (bool, optional): parameter which prints messages on the standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy, test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching RMS Prop training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        S = [torch.ones(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()  # re-init the gradients
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        S[i] = gamma * S[i] + (1-gamma) * param.grad.pow(2)
                        param -= torch.mul(
                            alpha/(epsilon + torch.sqrt(S[i])), param.grad)
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = torch.mean(((hat_y > 0.5).float() == y_train).float())
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = torch.mean(
                    ((output_test > 0.5).float() == y_test).float())
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, loss))
        if verbose:
            print('Loss test = {}, Accuracy test = {}'.format(lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_ADD(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs, batch_size,
            batches_per_epoch, rho, epsilon=1e-8, verbose=False):
        """AdaDelta optimization algorithm

        Args:
            Loss (torch.nn): loss function
            X_train (torch.tensor): train predictors
            y_train (torch.tensor): train target
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            batch_size (int): batch size
            batches_per_epoch (int): number of batch per epoch
            rho (float): decay factor
            epsilon (float, optional): stabilization parameter. Defaults to 1e-8
            verbose (bool, optional): parameter which prints messages on the standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy, test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching Ada Delta training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        S = [torch.zeros(param.shape) for param in self.parameters()]
        U = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()  # re-init the gradients
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        S[i] = rho * S[i] + (1 - rho) * param.grad.pow(2)
                        D_param = - torch.mul(
                            (epsilon + torch.sqrt(U[i]))/(
                                epsilon + torch.sqrt(S[i])), param.grad)
                        U[i] = rho * U[i] + (1 - rho) * D_param.pow(2)
                        param += D_param
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = torch.mean(((hat_y > 0.5).float() == y_train).float())
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = torch.mean(
                    ((output_test > 0.5).float() == y_test).float())
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, loss))

        if verbose:
            print('Loss test = {}, Accuracy test = {}'.format(lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_ADAM(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha, beta1, beta2,
            epsilon=1e-8, verbose=False):
        """Adam optimization algorithm

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
            beta2 (float): exponential decay rate for the second moment estimate
            epsilon (float, optional): stabilization parameter. Defaults to 1e-8
            verbose (bool, optional): parameter which prints messages on the standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy, test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching Adam training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        V = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()  # re-init the gradients
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        M[i] = beta1 * M[i] + (1-beta1) * param.grad
                        V[i] = beta2 * V[i] + (1-beta2) * param.grad.pow(2)
                        M_hat = (1/(1-beta1**(T+1))) * M[i]
                        V_hat = (1/(1-beta2**(T+1))) * V[i]
                        param -= alpha / (torch.sqrt(V_hat) + epsilon) * M_hat
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = torch.mean(((hat_y > 0.5).float() == y_train).float())
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = torch.mean(
                    ((output_test > 0.5).float() == y_test).float())
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, loss))
        if verbose:
            print('Loss test = {}, Accuracy test = {}'.format(lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_AMS(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha, beta1, beta2,
            epsilon=1e-8, verbose=False):
        """_summary_

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
            beta2 (float): exponential decay rate for the second moment estimate
            epsilon (float, optional): stabilization parameter. Defaults to 1e-8
            verbose (bool, optional): parameter which prints messages on the standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy, test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching AMS Grad training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        V = [torch.zeros(param.shape) for param in self.parameters()]
        V_hat = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()  # re-init the gradients
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        M[i] = beta1 * M[i] + (1-beta1) * param.grad
                        V[i] = beta2 * V[i] + (1-beta2) * param.grad.pow(2)
                        V_hat[i] = torch.maximum(V_hat[i], V[i])
                        param -= alpha / (torch.sqrt(V_hat[i]) + epsilon)*M[i]
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = torch.mean(((hat_y > 0.5).float() == y_train).float())
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = torch.mean(
                    ((output_test > 0.5).float() == y_test).float())
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, loss))
        if verbose:
            print('Loss test = {}, Accuracy test = {}'.format(lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_NADAM(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha, mu, nu,
            epsilon=1e-8, verbose=False):
        """NADAM optimization algorithm

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
            mu (float): exponential decay rate for the first moment estimate
            nu (float): exponential decay rate for the second moment estimate
            epsilon (float, optional): stabilization parameter. Defaults to 1e-8
            verbose (bool, optional): parameter which prints messages on the standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy, test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching Nadam training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        N = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()  # re-init the gradients
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, yb)  # Forward pass : Compute the loss
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
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = torch.mean(((hat_y > 0.5).float() == y_train).float())
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = torch.mean(
                    ((output_test > 0.5).float() == y_test).float())
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, loss))
        if verbose:
            print('Loss test = {}, Accuracy test = {}'.format(lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_ADAMAX(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha, beta1, beta2, lambd,
            epsilon=1e-8, verbose=False):
        """Adamax optimization algorithm

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
            beta2 (float): exponential decay rate for the second moment estimate
            lambd (float): regularization term
            epsilon (float, optional): stabilization parameter. Defaults to 1e-8
            verbose (bool, optional): parameter which prints messages on the standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy, test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching Adamax training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        U = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()  # re-init the gradients
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, yb)  # Forward pass : Compute the loss
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        g = param.grad
                        if lambd != 0:
                            g = g + lambd * param
                        M[i] = beta1 * M[i] + (1-beta1) * param.grad
                        U[i] = torch.maximum(beta2*U[i], torch.abs(g)+epsilon)
                        param -= alpha / ((1-beta1**(T+1))*U[i]) * M[i]
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = torch.mean(((hat_y > 0.5).float() == y_train).float())
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = torch.mean(
                    ((output_test > 0.5).float() == y_test).float())
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, loss))
        if verbose:
            print('Loss test = {}, Accuracy test = {}'.format(lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test
