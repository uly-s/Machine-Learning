from keras import backend as K
from keras.engine.topology import Layer
from keras.models import Model, Sequential
from keras.layers import Input, Multiply, Add, Dense, Lambda

def NLL(y_true, y_pred):
    LH = K.tf.distributions.Bernoulli(probs=y_pred)
    return -K.sum(LH.log_prob(y_true), axis=-1)


class KL(Layer):
    def __init__(self, *args, **kwargs):
        super(KL, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log = inputs
        batch = -0.5 * K.sum(1 + log - K.square(mu) - K.exp(log), axis=-1)
        self.add_loss(K.mean(batch), inputs=inputs)

        return inputs

def VariationalAutoEncoder(input, intermediate, latent):

    ### ENCODER ###

    x = Input(shape=(input,))

    h = Dense(intermediate, activation='relu')(x)

    ### MEAN AND LOG OF VARIANCE

    mu = Dense(latent)(h)

    log = Dense(latent)(h)

    ### STANDARD DEVIATION

    sigma = Lambda(lambda t: K.exp(0.5*t))(log)

    ### KL DIVERGENCE LAYER, TAKES MEAN AND LOG VARIANCE

    mu, log = KL()([mu, log])

    ### EPSILON FOR GAUSSIAN NOISE TERM

    eps = Input(tensor=K.random_normal(shape=(K.shape(x)[0], latent)))

    ### MULTIPLY STD DEV AND NOISE

    z_eps = Multiply()([sigma, eps])

    ### OUTPUT  OF ENCODER, MEAN PLUS NOISY SIGMA

    z = Add()([mu, z_eps])

    ### DECODER ###

    decoder = Sequential([Dense(intermediate, input_dim=latent, activation='relu'),
                          Dense(input, activation='sigmoid')])(z)

    ### MODEL ###

    VAE = Model(inputs=[x, eps], outputs=decoder)
    VAE.compile(optimizer='rmsprop', loss=NLL)


    return VAE



















