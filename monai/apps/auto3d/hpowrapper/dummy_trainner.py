class trainner():
    def __init__(self, config={}):
        self.config = config

    ## required
    def update(self, params):
        self.config.update(params)

    # required
    def train(self):
        acc = self.config['lr'] ** 2
        return acc
