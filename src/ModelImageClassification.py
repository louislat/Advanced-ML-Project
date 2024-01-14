# ModelImageClassification

# import numpy as np
import torch
from torcheval.metrics.functional import multiclass_accuracy
from numpy.random import choice


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=(5, 5), padding='same')
        self.relu1 = torch.nn.ReLU()
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=(5, 5), padding='same')
        self.relu2 = torch.nn.ReLU()
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=2)
        self.flat = torch.nn.Flatten()
        self.mlp1 = torch.nn.Linear(in_features=784, out_features=120)
        self.mlp2 = torch.nn.Linear(in_features=120, out_features=84)
        self.mlp3 = torch.nn.Linear(in_features=84, out_features=10)

    def forward(self, inp):
        x = self.conv1(inp)
        x = self.relu1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool2(x)
        x = self.flat(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        return (x)

    def training_SGD(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha, verbose=False):
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching SDG training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)
                loss = Loss(hat_yb, yb)  # Forward pass : Compute the loss
                self.zero_grad()
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for param in self.parameters():  # Parameters' update
                        param -= alpha * param.grad
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = multiclass_accuracy(hat_y, y_train)
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = multiclass_accuracy(output_test, y_test)
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print((
                    "".join([
                        'Epoch {} / {} : Loss train = {} |',
                        'Accu train = {} | Loss test = {} |',
                        'Accu test = {}'])).format(
                            T+1, n_epochs, loss, acc, lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_Mom(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha, beta, verbose=False):
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
                hat_yb = self.forward(Xb)
                loss = Loss(hat_yb, yb)  # Forward pass : Compute the loss
                self.zero_grad()
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        Momentums[i] = beta*Momentums[i] + (1-beta)*param.grad
                        param -= alpha * Momentums[i]
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = multiclass_accuracy(hat_y, y_train)
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = multiclass_accuracy(output_test, y_test)
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print((
                    "".join([
                        'Epoch {} / {} : Loss train = {} |',
                        'Accu train = {} | Loss test = {} |',
                        'Accu test = {}'])).format(
                            T+1, n_epochs, loss, acc, lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_NAG(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha, beta, verbose=False):
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
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        param -= alpha*beta*Momentums[i]
                hat_yb = self.forward(Xb)
                loss = Loss(hat_yb, yb)
                loss.backward()
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        Momentums[i] = beta*Momentums[i] + (1-beta)*param.grad
                        param -= alpha*(1-beta)*param.grad
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = multiclass_accuracy(hat_y, y_train)
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = multiclass_accuracy(output_test, y_test)
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print((
                    "".join([
                        'Epoch {} / {} : Loss train = {} |',
                        'Accu train = {} | Loss test = {} |',
                        'Accu test = {}'])).format(
                            T+1, n_epochs, loss, acc, lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_ADG(
        self, Loss, X_train, y_train, X_test, y_test,
            n_epochs, batch_size, batches_per_epoch,
            alpha=0.01, epsilon=1e-8, verbose=False):
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
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)
                loss = Loss(hat_yb, yb)
                loss.backward()
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        S[i] += param.grad.pow(2)
                        param -= torch.mul(
                            alpha/(epsilon + torch.sqrt(S[i])), param.grad)
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = multiclass_accuracy(hat_y, y_train)
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = multiclass_accuracy(output_test, y_test)
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print((
                    "".join([
                        'Epoch {} / {} : Loss train = {} |',
                        'Accu train = {} | Loss test = {} |',
                        'Accu test = {}'])).format(
                            T+1, n_epochs, loss, acc, lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_RMS(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha=0.01,
            gamma=0.9, epsilon=1e-8, verbose=False):
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
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)
                loss = Loss(hat_yb, yb)
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        S[i] = gamma * S[i] + (1-gamma) * param.grad.pow(2)
                        param -= torch.mul(
                            alpha/(epsilon + torch.sqrt(S[i])), param.grad)
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = multiclass_accuracy(hat_y, y_train)
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = multiclass_accuracy(output_test, y_test)
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print((
                    "".join([
                        'Epoch {} / {} : Loss train = {} |',
                        'Accu train = {} | Loss test = {} |',
                        'Accu test = {}'])).format(
                            T+1, n_epochs, loss, acc, lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_ADD(
        self, Loss, X_train, y_train, X_test, y_test,
            n_epochs, batch_size, batches_per_epoch,
            rho, epsilon=1e-8, verbose=False):
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching AdaDelta training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        S = [torch.zeros(param.shape) for param in self.parameters()]  # RMS g
        U = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)
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
                acc = multiclass_accuracy(hat_y, y_train)
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = multiclass_accuracy(output_test, y_test)
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print((
                    "".join([
                        'Epoch {} / {} : Loss train = {} |',
                        'Accu train = {} | Loss test = {} |',
                        'Accu test = {}'])).format(
                            T+1, n_epochs, loss, acc, lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_ADAM(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha, beta1, beta2,
            epsilon=1e-8, verbose=False):
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching Adam training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        # Momentums
        M = [torch.zeros(param.shape) for param in self.parameters()]
        # Adaptative terms
        V = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)
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
                acc = multiclass_accuracy(hat_y, y_train)
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = multiclass_accuracy(output_test, y_test)
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print((
                    "".join([
                        'Epoch {} / {} : Loss train = {} |',
                        'Accu train = {} | Loss test = {} |',
                        'Accu test = {}'])).format(
                            T+1, n_epochs, loss, acc, lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_AMS(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs, batch_size,
            batches_per_epoch, alpha, beta1, beta2,
            epsilon=1e-8, verbose=False):
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching AMS Grad training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        # Momentums
        M = [torch.zeros(param.shape) for param in self.parameters()]
        # Adaptative terms
        V = [torch.zeros(param.shape) for param in self.parameters()]
        V_hat = [torch.zeros(param.shape) for param in self.parameters()]
        for T in range(n_epochs):
            for _ in range(batches_per_epoch):
                self.zero_grad()  # re-init the gradients
                batch_indexes = choice(
                    range(y_train.shape[0]), size=batch_size, replace=False)
                Xb, yb = X_train[batch_indexes], y_train[batch_indexes]
                hat_yb = self.forward(Xb)  # Forward pass
                loss = Loss(hat_yb, yb)  # Forward pass
                loss.backward()  # Backpropagation
                with torch.no_grad():
                    for i, param in enumerate(self.parameters()):
                        M[i] = beta1 * M[i] + (1-beta1) * param.grad
                        V[i] = beta2 * V[i] + (1-beta2) * param.grad.pow(2)
                        V_hat[i] = torch.maximum(V_hat[i], V[i])
                        param -= alpha / (torch.sqrt(V_hat[i]) + epsilon)*M[i]
            with torch.no_grad():
                hat_y = self.forward(X_train)
                acc = multiclass_accuracy(hat_y, y_train)
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = multiclass_accuracy(output_test, y_test)
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print((
                    "".join([
                        'Epoch {} / {} : Loss train = {} |',
                        'Accu train = {} | Loss test = {} |',
                        'Accu test = {}'])).format(
                            T+1, n_epochs, loss, acc, lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_NADAM(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs,
            batch_size, batches_per_epoch, alpha, mu, nu,
            epsilon=1e-8, verbose=False):
        Accuracies = []
        Losses = []
        Accuracies_test = []
        Losses_test = []
        if verbose:
            print('Launching Nadam training of model')
            print('Number of parameters : {}'.format(
                sum(p.numel() for p in self.parameters())))
        # Momentums
        M = [torch.zeros(param.shape) for param in self.parameters()]
        # Adaptative terms
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
                acc = multiclass_accuracy(hat_y, y_train)
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = multiclass_accuracy(output_test, y_test)
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print((
                    "".join([
                        'Epoch {} / {} : Loss train = {} |',
                        'Accu train = {} | Loss test = {} |',
                        'Accu test = {}'])).format(
                            T+1, n_epochs, loss, acc, lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test

    def training_ADAMAX(
        self, Loss, X_train, y_train, X_test, y_test, n_epochs, batch_size,
            batches_per_epoch, alpha, beta1, beta2, lambd,
            epsilon=1e-8, verbose=False):
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
                acc = multiclass_accuracy(hat_y, y_train)
                loss = Loss(hat_y, y_train)
                output_test = self.forward(X_test)
                lt = Loss(output_test, y_test)
                at = multiclass_accuracy(output_test, y_test)
            Accuracies.append(acc)
            Losses.append(loss)
            Accuracies_test.append(at)
            Losses_test.append(lt)
            if verbose and T % 10 == 0:
                print((
                    "".join([
                        'Epoch {} / {} : Loss train = {} |',
                        'Accu train = {} | Loss test = {} |',
                        'Accu test = {}'])).format(
                            T+1, n_epochs, loss, acc, lt, at))
        return Losses, Accuracies, Losses_test, Accuracies_test
