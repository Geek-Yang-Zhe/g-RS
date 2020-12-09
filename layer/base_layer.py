class Layer(object):
    def call(self, inputs):
        return inputs

    def __call__(self, inputs):
        return self.call(inputs)