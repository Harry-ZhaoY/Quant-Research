from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, LSTM, Bidirectional, Concatenate
from keras.models import Model
from keras import layers
from tensorflow.keras import backend as K

class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        e = K.dot(inputs, self.W)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        context = inputs * K.expand_dims(alpha, -1)
        context = K.sum(context, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class CNNBiLSTMAttentionModel(Model):
    def __init__(self, input_shape_cnn, input_shape_lstm, **kwargs):
        super(CNNBiLSTMAttentionModel, self).__init__(**kwargs)
        # CNN layers
        self.conv1 = Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.flatten_cnn = Flatten()
        self.dense_cnn = Dense(64, activation='relu')

        # LSTM layers
        self.bilstm = Bidirectional(LSTM(64, return_sequences=True))
        self.attention = AttentionLayer()

        # Dense layers for final prediction
        self.dense_final = Dense(64, activation='relu')
        self.final_output = Dense(1)

        # Input layers
        self.input_cnn = Input(shape=input_shape_cnn, name='input_cnn')
        self.input_lstm = Input(shape=input_shape_lstm, name='input_lstm')

    def call(self, inputs):
        # Split inputs
        input_cnn, input_lstm = inputs

        # CNN part
        x_cnn = self.conv1(input_cnn)
        x_cnn = self.pool1(x_cnn)
        x_cnn = self.flatten_cnn(x_cnn)
        x_cnn = self.dense_cnn(x_cnn)

        # LSTM part
        x_lstm = self.bilstm(input_lstm)
        x_lstm = self.attention(x_lstm)
        x_lstm = layers.Flatten()(x_lstm)

        # Combining both parts
        x = Concatenate(axis=-1)([x_cnn, x_lstm])
        x = self.dense_final(x)
        final_output = self.final_output(x)
        return final_output

    def model(self):
        return Model(inputs=[self.input_cnn, self.input_lstm], outputs=self.call([self.input_cnn, self.input_lstm]))


# # Now you can create an instance of this class and compile it:
# SEQUENCE_LENGTH = 60

# model = CNNBiLSTMAttentionModel((SEQUENCE_LENGTH, 5, 1), (SEQUENCE_LENGTH, 5))
# model.compile(optimizer='adam', loss='mean_squared_error')

# # To see the summary, you can call the model method:
# model.model().summary()
