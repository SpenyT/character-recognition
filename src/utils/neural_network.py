class NeuralNetwork:
    def __init__(self, loss=None, loss_prime=None):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime

    def add(self, layer):
        self.layers.append(layer)
    
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def forward_prop(self, input_data):
        for layer in self.layers:
            input_data = layer.forward_prop(input_data)
        return input_data
    
    def backward_prop(self, loss_grad, learning_rate):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_prop(loss_grad, learning_rate)

    # def train(self, x_train, y_train, epochs, learning_rate, print_output: True):
    #     for epoch in range(epochs):
    #         err = 0
    #         for x, y in zip(x_train, y_train):
    #             output = self.forward_prop(x)
    #             err += self.loss(y, output)
    #             grad = self.loss_prime(y, output, learning_rate)
    #         err /= len(x_train)
    #         print(f'Epoch {epoch+1}/epochs    Error={err:.6f}')
    

    # def predict(self, x_test):
    #     results = []
    #     for x in x_test:
    #         output = self.forward_prop(x)
    #         results.append(output)
    #     return results
