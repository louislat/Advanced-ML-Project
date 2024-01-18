import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    """LSTM model for text classification
    """
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        """Class constructor

        Args:
            input_dim (int): number of words in the vocabulary
            embedding_dim (int): size of the embedding
            hidden_dim (int): number of hidden neurons
            output_dim (int): dimension of the ouput
        """
        self.hidden_dim = hidden_dim
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        final_output = self.fc(lstm_out[:,-1])
        return final_output

    def training_SGD(
        self, Loss, train_loader, X_test, y_test, n_epochs, alpha, verbose=False):
        """SGD optimization algorithm

        Args:
            Loss (torch.nn): loss function
            train_loader (torch.utils.data.DataLoader): train DataLoader
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float): learning rate
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy,
            test loss, test accuracy
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
            for x_train , y_train in train_loader:
                hat_yb = self.forward(x_train)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, y_train.unsqueeze(1).float())  # Forward pass : Compute the loss
                acc = torch.mean(((F.sigmoid(hat_yb) > 0.5).squeeze().float() == y_train).float())
                self.zero_grad()  # re-init the gradients
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for param in self.parameters():  # Parameters' update
                        param -= alpha * param.grad

            with torch.no_grad():
                loss_t = Loss(self.forward(X_test), y_test.unsqueeze(1).float())
                acc_t = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_test).float())

            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(acc_t)
            Losses_test.append(loss_t)

            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, Losses[-1]))
            
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_Mom(
        self, Loss, train_loader, X_test, y_test, n_epochs, alpha, beta, verbose=False):
        """Momentum optimization algorithm

        Args:
            Loss (torch.nn): loss function
            train_loader (torch.utils.data.DataLoader): train DataLoader
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float): learning rate
            beta (float): weightage that is going to assign to the past values
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy,
            test loss, test accuracy
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
            for x_train , y_train in train_loader:
                hat_yb = self.forward(x_train)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, y_train.unsqueeze(1).float())  # Forward pass : Compute the loss
                acc = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_train).float())
                self.zero_grad()  # re-init the gradients
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        Momentums[i] = beta*Momentums[i] + (1-beta)*param.grad
                        param -= alpha * Momentums[i]

            with torch.no_grad():
                loss_t = Loss(self.forward(X_test), y_test.unsqueeze(1).float())
                acc_t = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_test).float())

            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(acc_t)
            Losses_test.append(loss_t)

            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, Losses[-1]))
            
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_NAG(
        self, Loss, train_loader, X_test, y_test, n_epochs, alpha, beta, verbose=False):
        """NAG optimization algorithm

        Args:
            Loss (torch.nn): loss function
            train_loader (torch.utils.data.DataLoader): train DataLoader
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float): learning rate
            beta (float): weightage that is going to assign to the past values
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy,
            test loss, test accuracy
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
            for x_train , y_train in train_loader:
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        param -= alpha*beta*Momentums[i]
                hat_yb = self.forward(x_train)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, y_train.unsqueeze(1).float())  # Forward pass : Compute the loss
                acc = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_train).float())
                self.zero_grad()  # re-init the gradients
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        Momentums[i] = beta*Momentums[i] + (1-beta)*param.grad
                        param -= alpha*(1-beta)*param.grad
            with torch.no_grad():
                loss_t = Loss(self.forward(X_test), y_test.unsqueeze(1).float())
                acc_t = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_test).float())

            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(acc_t)
            Losses_test.append(loss_t)

            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, Losses[-1]))
            
        return Losses, Accuracies, Losses_test, Accuracies_test
    
    def training_ADG(
        self, Loss, train_loader, X_test, y_test, n_epochs, alpha=0.01, epsilon=1e-8, verbose=False):
        """ADG optimization algorithm

        Args:
            Loss (torch.nn): loss function
            train_loader (torch.utils.data.DataLoader): train DataLoader
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float, optional): learning rate. Defaults to 0.01
            epsilon (float, optional): stabilization parameter. Defaults to
            1e-8
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy,
            test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching AGD training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        S = [torch.ones(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for x_train , y_train in train_loader:
                hat_yb = self.forward(x_train)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, y_train.unsqueeze(1).float())  # Forward pass : Compute the loss
                acc = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_train).float())
                self.zero_grad()  # re-init the gradients
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        S[i] += param.grad.pow(2)
                        param -= torch.mul(alpha/(epsilon + torch.sqrt(S[i])), param.grad)
            with torch.no_grad():
                loss_t = Loss(self.forward(X_test), y_test.unsqueeze(1).float())
                acc_t = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_test).float())

            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(acc_t)
            Losses_test.append(loss_t)

            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, Losses[-1]))
            
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_RMS(
        self, Loss, train_loader, X_test, y_test, n_epochs, alpha=0.01, gamma=0.9, epsilon=1e-8, verbose=False):
        """RMS Prop optimization algorithm

        Args:
            Loss (torch.nn): loss function
            train_loader (torch.utils.data.DataLoader): train DataLoader
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
            Tuple[list, list, list, list]: train loss, train accuracy,
            test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching RMS training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        S = [torch.ones(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for x_train , y_train in train_loader:
                hat_yb = self.forward(x_train)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, y_train.unsqueeze(1).float())  # Forward pass : Compute the loss
                acc = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_train).float())
                self.zero_grad()  # re-init the gradients
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        S[i] = gamma * S[i] + (1-gamma) * param.grad.pow(2)
                        param -= torch.mul(
                            alpha/(epsilon + torch.sqrt(S[i])), param.grad)
            with torch.no_grad():
                loss_t = Loss(self.forward(X_test), y_test.unsqueeze(1).float())
                acc_t = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_test).float())

            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(acc_t)
            Losses_test.append(loss_t)

            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, Losses[-1]))
            
        return Losses, Accuracies, Losses_test, Accuracies_test
    
    def training_ADAM(
        self, Loss, train_loader, X_test, y_test, n_epochs, alpha, beta1, beta2 , epsilon=1e-8, verbose=False):
        """ADAM optimization algorithm

        Args:
            Loss (torch.nn): loss function
            train_loader (torch.utils.data.DataLoader): train DataLoader
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
            Tuple[list, list, list, list]: train loss, train accuracy,
            test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching ADAM training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        V = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for x_train , y_train in train_loader:
                hat_yb = self.forward(x_train)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, y_train.unsqueeze(1).float())  # Forward pass : Compute the loss
                acc = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_train).float())
                self.zero_grad()  # re-init the gradients
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        M[i] = beta1 * M[i] + (1-beta1) * param.grad
                        V[i] = beta2 * V[i] + (1-beta2) * param.grad.pow(2)
                        M_hat = (1/(1-beta1**(T+1))) * M[i]
                        V_hat = (1/(1-beta2**(T+1))) * V[i]
                        param -= alpha / (torch.sqrt(V_hat) + epsilon) * M_hat
            with torch.no_grad():
                loss_t = Loss(self.forward(X_test), y_test.unsqueeze(1).float())
                acc_t = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_test).float())

            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(acc_t)
            Losses_test.append(loss_t)

            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, Losses[-1]))
            
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_AMS(
        self, Loss, train_loader, X_test, y_test, n_epochs, alpha, beta1, beta2 , epsilon=1e-8, verbose=False):
        """AMS Grad optimization algorithm

        Args:
            Loss (torch.nn): loss function
            train_loader (torch.utils.data.DataLoader): train DataLoader
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
            Tuple[list, list, list, list]: train loss, train accuracy,
            test loss, test accuracy
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
            for x_train , y_train in train_loader:
                hat_yb = self.forward(x_train)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, y_train.unsqueeze(1).float())  # Forward pass : Compute the loss
                acc = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_train).float())
                self.zero_grad()  # re-init the gradients
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        M[i] = beta1 * M[i] + (1-beta1) * param.grad
                        V[i] = beta2 * V[i] + (1-beta2) * param.grad.pow(2)
                        V_hat[i] = torch.maximum(V_hat[i], V[i])
                        param -= alpha / (torch.sqrt(V_hat[i]) + epsilon)*M[i]
            with torch.no_grad():
                loss_t = Loss(self.forward(X_test), y_test.unsqueeze(1).float())
                acc_t = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_test).float())

            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(acc_t)
            Losses_test.append(loss_t)

            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, Losses[-1]))
            
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_NADAM(
        self, Loss, train_loader, X_test, y_test, n_epochs, alpha, mu, nu , epsilon=1e-8, verbose=False):
        """NADAM optimization algorithm

        Args:
            Loss (torch.nn): loss function
            train_loader (torch.utils.data.DataLoader): train DataLoader
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float): learning rate
            mu (float): exponential decay rate for the first moment estimate
            nu (float): exponential decay rate for the second moment estimate
            epsilon (float, optional): stabilization parameter. Defaults to
            1e-8
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy,
            test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching NADAM training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        N = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for x_train , y_train in train_loader:
                hat_yb = self.forward(x_train)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, y_train.unsqueeze(1).float())  # Forward pass : Compute the loss
                acc = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_train).float())
                self.zero_grad()  # re-init the gradients
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
                loss_t = Loss(self.forward(X_test), y_test.unsqueeze(1).float())
                acc_t = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_test).float())

            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(acc_t)
            Losses_test.append(loss_t)

            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, Losses[-1]))
            
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_ADAMAX(
        self, Loss, train_loader, X_test, y_test, n_epochs, alpha, beta1, beta2, lambd, epsilon=1e-8, verbose=False):
        """ADAMAX optimization algorithm

        Args:
            Loss (torch.nn): loss function
            train_loader (torch.utils.data.DataLoader): train DataLoader
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
            alpha (float): learning rate
            beta1 (float): exponential decay rate for the first moment estimate
            beta2 (float): exponential decay rate for the second
            moment estimate
            lambd (float): regularization term
            epsilon (float, optional): stabilization parameter. Defaults to
            1e-8
            verbose (bool, optional): parameter which prints messages on the
            standard output. Defaults to False.

        Returns:
            Tuple[list, list, list, list]: train loss, train accuracy,
            test loss, test accuracy
        """
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching ADAMAX training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        U = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for x_train , y_train in train_loader:
                hat_yb = self.forward(x_train)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, y_train.unsqueeze(1).float())  # Forward pass : Compute the loss
                acc = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_train).float())
                self.zero_grad()  # re-init the gradients
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
                loss_t = Loss(self.forward(X_test), y_test.unsqueeze(1).float())
                acc_t = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_test).float())

            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(acc_t)
            Losses_test.append(loss_t)

            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T+1, n_epochs, Losses[-1]))
            
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_NOS(
        self, Loss, train_loader, X_test, y_test, n_epochs, alpha, beta1, gamma, epsilon=1e-8, verbose=False):
        """Nostalgic Adam optimization algorithm

        Args:
            Loss (torch.nn): loss function
            train_loader (torch.utils.data.DataLoader): train DataLoader
            X_test (torch.tensor): test predictors
            y_test (torch.tensor): test target
            n_epochs (int): number of epochs
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
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching Nostalgic Adam training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        M = [torch.zeros(param.shape) for param in self.parameters()]
        V = [torch.zeros(param.shape) for param in self.parameters()]
        B1, B2 = 0, 1
        for T in range(1, n_epochs+1):
            for x_train , y_train in train_loader:
                hat_yb = self.forward(x_train)  # Forward pass: Compute predicted y
                loss = Loss(hat_yb, y_train.unsqueeze(1).float())  # Forward pass : Compute the loss
                acc = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_train).float())
                self.zero_grad()  # re-init the gradients
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
            with torch.no_grad():
                loss_t = Loss(self.forward(X_test), y_test.unsqueeze(1).float())
                acc_t = torch.mean(((F.sigmoid(hat_yb) > 0.5).float() == y_test).float())

            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(acc_t)
            Losses_test.append(loss_t)

            if verbose and T % 10 == 0:
                print('Epoch {} / {} : Loss = {}'.format(T, n_epochs, Losses[-1]))
            
        return Losses, Accuracies, Losses_test, Accuracies_test