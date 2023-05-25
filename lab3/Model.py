import numpy as np
import math


class Model:

    def __init__(self, inp, activation, derivative, seed):
        np.random.seed(seed)
        self._inp = inp
        self._layers = []
        self._activation = activation
        self._derivative = derivative

    def add_layer(self, hidden, out):
        k = 1 / math.sqrt(hidden)
        prev_layer_neurons = self._inp
        if len(self._layers) > 0:
            prev_layer_neurons = self._layers[-1]['neurons']
        self._layers.append({
            'neurons': out,
            'inp_w': np.random.rand(prev_layer_neurons, hidden) * 2 * k - k,
            'h_w': np.random.rand(hidden, hidden) * 2 * k - k,
            'h_b': np.random.rand(1, hidden) * 2 * k - k,
            'out_w': np.random.rand(hidden, out) * 2 * k - k,
            'out_b': np.random.rand(1, out) * 2 * k - k
        })

    def step_forward(self, x):
        hidden_list = []
        output_list = []
        for layer in self._layers:
            hidden = np.zeros((x.shape[0], layer['inp_w'].shape[1]))
            output = np.zeros((x.shape[0], layer['out_w'].shape[1]))
            for j in range(x.shape[0]):
                input_x = np.matmul(x[j, :][np.newaxis, :], layer['inp_w'])
                hidden_x = np.matmul(input_x + hidden[max(j-1, 0), :][np.newaxis, :], layer['h_w'] + layer['h_b'])
                hidden_x = self._activation(hidden_x)
                hidden[j, :] = hidden_x
                output_x = np.matmul(hidden_x, layer['out_w'] + layer['out_b'])
                output[j, :] = output_x
            hidden_list.append(hidden)
            output_list.append(output)
        return hidden_list, output_list[-1]

    def step_backward(self, x, a, error, hidden_list):
        for i in range(len(self._layers)):
            layer = self._layers[i]
            hidden = hidden_list[i]
            next_h_g = None
            inp_w_g = 0
            h_w_g = 0
            h_b_g = 0
            out_w_g = 0
            out_b_g = 0
            for j in range(x.shape[0] - 1, -1, -1):
                out_grad = error[j, :][np.newaxis, :]
                out_w_g += np.matmul(hidden[j, :][:, np.newaxis], out_grad)
                out_b_g += out_grad
                h_grad = np.matmul(out_grad, layer['out_w'].T)
                if j < x.shape[0] - 1:
                    hh_grad = np.matmul(next_h_g, layer['h_w'].T)
                    h_grad += hh_grad
                derivative = self._derivative(hidden[j][np.newaxis, :])
                h_grad = np.multiply(h_grad, derivative)
                next_h_g = h_grad.copy()
                if j > 0:
                    h_w_g += np.matmul(hidden[j - 1][:, np.newaxis], h_grad)
                    h_b_g += h_grad
                inp_w_g += np.matmul(x[j, :][:, np.newaxis], h_grad)
            a /= x.shape[0]
            self._layers[i]['inp_w'] -= inp_w_g * a
            self._layers[i]['h_w'] -= h_w_g * a
            self._layers[i]['h_b'] -= h_b_g * a
            self._layers[i]['out_w'] -= out_w_g * a
            self._layers[i]['out_b'] -= out_b_g * a

    def fit(self, x, y, x_test, y_test, loss_f, sl, a, epochs):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for j in range(x.shape[0] - sl):
                seq_x = x[j:(j + sl), ]
                seq_y = y[j:(j + sl), ]
                hidden_list, output_list = self.step_forward(seq_x)
                error = output_list - seq_y
                self.step_backward(seq_x, a, error, hidden_list)
                epoch_loss += loss_f(seq_y, output_list)
            valid_loss = 0
            for j in range(x_test.shape[0] - sl):
                seq_x = x_test[j:(j + sl), ]
                seq_y = y_test[j:(j + sl), ]
                _, output_list = self.step_forward(seq_x)
                valid_loss += loss_f(seq_y, output_list)

            loss = epoch_loss / len(x)
            test_loss = valid_loss / len(x_test)
            print(f"Epoch: {epoch} train loss {loss} valid loss {test_loss}")
            losses.append((loss, test_loss))
        return losses

    def predict(self, x):
        return self.step_forward(x)[1]
