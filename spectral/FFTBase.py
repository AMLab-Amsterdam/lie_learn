
class FFTBase(object):

    def __init__(self):
        pass

    def analyze(self, f):
        raise NotImplementedError('FFTBase.analyze should be implemented in subclass')

    def synthesize(self, f_hat):
        raise NotImplementedError('FFTBase.synthesize should be implemented in subclass')
