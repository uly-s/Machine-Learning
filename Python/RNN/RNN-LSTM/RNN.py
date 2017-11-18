import numpy as np





class RNN:

    def __init__(self, cell_count, xdim):
        """Pass the memory cell count and the input dimension,
            parameters are as follows:
            g = input nodes
            i = input gate
            f = forget gate
            o = output gate
            Wvar and Bvar are weights and biases
            dvar is delta variable, concat length
            is cell count + xdim"""

        self.cell_count = cell_count
        self.xdim = xdim
        self.concat_len = cell_count + xdim

        # weights
        self.Wg = np.random.rand(cell_count, self.concat_len) * 0.1
        self.Wi = np.random.rand(cell_count, self.concat_len) * 0.1
        self.Wf = np.random.rand(cell_count, self.concat_len) * 0.1
        self.Wo = np.random.rand(cell_count, self.concat_len) * 0.1

        # biases
        self.bg = np.random.randn(cell_count) * 0.1
        self.bi = np.random.randn(cell_count) * 0.1
        self.bf = np.random.randn(cell_count) * 0.1
        self.bo = np.random.randn(cell_count) * 0.1

        # derivative of parameters
        # weights
        self.dWg = np.zeros((cell_count, self.concat_len))
        self.dWi = np.zeros((cell_count, self.concat_len))
        self.dWf = np.zeros((cell_count, self.concat_len))
        self.dWo = np.zeros((cell_count, self.concat_len))

        # bias
        self.dbg = np.zeros(cell_count)
        self.dbi = np.zeros(cell_count)
        self.dbf = np.zeros(cell_count)
        self.dbo = np.zeros(cell_count)


    def update(self, LR):
        """Apply derivative to parameters, pass the learning rate"""

        # update weights
        self.Wg -= LR * self.dWg
        self.Wi -= LR * self.dWi
        self.Wf -= LR * self.dWf
        self.Wo -= LR * self.dWo

        # update bias
        self.bg -= LR * self.dbg
        self.bi -= LR * self.dbi
        self.bf -= LR * self.dbf
        self.bo -= LR * self.dbo

        # reset derivatives
        # weights
        self.dWg = np.zeros_like(self.Wg)
        self.dWi = np.zeros_like(self.Wi)
        self.dWf = np.zeros_like(self.Wf)
        self.dWo = np.zeros_like(self.Wo)

        # bias
        self.dbg = np.zeros_like(self.bg)
        self.dbi = np.zeros_like(self.bi)
        self.dbf = np.zeros_like(self.bf)
        self.dbo = np.zeros_like(self.bo)


