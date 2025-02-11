import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Reshape,
    UpSampling2D,
    Conv2DTranspose,
    MultiHeadAttention,
    LayerNormalization
)
from qkeras import QActivation, QConv2D, QDense, QDenseBatchnorm, quantized_bits


class TeacherAutoencoder:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="teacher_inputs_")
        x = Reshape((18, 14, 1), name="teacher_reshape")(inputs)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_1")(x)
        x = Activation("relu", name="teacher_relu_1")(x)
        x = AveragePooling2D((2, 2), name="teacher_pool_1")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_2")(x)
        x = Activation("relu", name="teacher_relu_2")(x)
        x = Flatten(name="teacher_flatten")(x)
        x = Dense(80, activation="relu", name="teacher_latent")(x)
        x = Dense(9 * 7 * 30, name="teacher_dense")(x)
        x = Reshape((9, 7, 30), name="teacher_reshape2")(x)
        x = Activation("relu", name="teacher_relu_3")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_3")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = UpSampling2D((2, 2), name="teacher_upsampling")(x)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_4")(x)
        x = Activation("relu", name="teacher_relu_5")(x)
        outputs = Conv2D(
            1,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            name="teacher_outputs",
        )(x)
        return Model(inputs, outputs, name="teacher")


class TeacherAutoencoderRevised:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="teacher_inputs_")
        x = Reshape((18, 14, 1), name="teacher_reshape")(inputs)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_1")(x)
        x = Activation("relu", name="teacher_relu_1")(x)
        x = AveragePooling2D((2, 2), name="teacher_pool_1")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_2")(x)
        x = Activation("relu", name="teacher_relu_2")(x)
        x = Flatten(name="teacher_flatten")(x)
        x = Dense(80, activation="relu", name="teacher_latent")(x)
        x = Dense(9 * 7 * 30, name="teacher_dense")(x)
        x = Reshape((9, 7, 30), name="teacher_reshape2")(x)
        x = Activation("relu", name="teacher_relu_3")(x)
        x = Conv2D(30, (3, 3), strides=1, padding="same", name="teacher_conv2d_3")(x)
        x = Activation("relu", name="teacher_relu_4")(x)
        x = Conv2DTranspose(30, (3, 3), strides=2, padding="same", name="teacher_conv_transpose")(x)
        x = Conv2D(20, (3, 3), strides=1, padding="same", name="teacher_conv2d_4")(x)
        x = Activation("relu", name="teacher_relu_5")(x)
        outputs = Conv2D(
            1,
            (3, 3),
            activation="relu",
            strides=1,
            padding="same",
            name="teacher_outputs",
        )(x)
        return Model(inputs, outputs, name="teacher-transpose")


class CicadaV1:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="inputs_")
        x = QDenseBatchnorm(
            16,
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0),
            name="dense1",
        )(inputs)
        x = QActivation("quantized_relu(10, 6)", name="relu1")(x)
        x = Dropout(1 / 8)(x)
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name="dense2",
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name="outputs")(x)
        return Model(inputs, outputs, name="cicada-v1")


class CicadaV2:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="inputs_")
        x = Reshape((18, 14, 1), name="reshape")(inputs)
        x = QConv2D(
            4,
            (2, 2),
            strides=2,
            padding="valid",
            use_bias=False,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            name="conv",
        )(x)
        x = QActivation("quantized_relu(10, 6)", name="relu0")(x)
        x = Flatten(name="flatten")(x)
        x = Dropout(1 / 9)(x)
        x = QDenseBatchnorm(
            16,
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0),
            name="dense1",
        )(x)
        x = QActivation("quantized_relu(10, 6)", name="relu1")(x)
        x = Dropout(1 / 8)(x)
        x = QDense(
            1,
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0),
            use_bias=False,
            name="dense2",
        )(x)
        outputs = QActivation("quantized_relu(16, 8)", name="outputs")(x)
        return Model(inputs, outputs, name="cicada-v2")

class CNN_Trial:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_trial(self, trial):
        return self.create(
            n_filters = trial.suggest_int('n_filters', 1, 8), 
            n_conv_layers = trial.suggest_int('n_conv_layers', 0, 3), 
            n_dense_units = trial.suggest_int('n_dense_units', 1, 32), 
            n_dense_layers = trial.suggest_int('n_dense_layers', 0, 8), 
            )

    def get_model(self, params):
        return self.create(
            n_filters = params['n_filters'], 
            n_conv_layers = params['n_conv_layers'], 
            n_dense_units = params['n_dense_units'], 
            n_dense_layers = params['n_dense_layers'], 
            )

    def create(self, n_filters, n_conv_layers, n_dense_units, n_dense_layers):

        # Input layer
        model = Sequential()
        model.add(Input(shape=(self.input_shape), name="input"))
        model.add(Reshape((18, 14, 1), name='reshape'))

        # Convolutional layers
        for i in range(n_conv_layers):
            model.add(QConv2D(
                n_filters, 
                (2, 2), 
                strides=2, 
                padding='valid', 
                use_bias=False, 
                kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0), 
                name=f'conv{i}',
            ))
            model.add(QActivation("quantized_relu(10, 6)", name=f'relu{i}'))
            model.add(Dropout(1/9))
        model.add(Flatten(name='flatten'))

        # Dense layers
        for i in range(n_dense_layers):
            model.add(QDenseBatchnorm(
                n_dense_units, 
                kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0),
                bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0), 
                name=f'dense{i}',
            ))
            model.add(QActivation("quantized_relu(10, 6)", name=f'relu{n_conv_layers+i}'))
            model.add(Dropout(1/8))

        # Output layer
        model.add(QDense(
            1, 
            kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0), 
            use_bias=False, 
            name='dense_output',
        ))
        model.add(QActivation("quantized_relu(16, 8)", name='output'))

        return model

class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size, n_hidden):
        super().__init__()
        self.patch_size = patch_size
        self.n_hidden = n_hidden
        self.flatten = Flatten()
        self.dense = QDense(
            n_hidden, 
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0), 
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0)
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        patches = tf.image.extract_patches(
            x, sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1], padding='VALID'
        )
        patches = self.flatten(patches)
        return self.dense(patches)
    
class ViT_Trial:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

    def get_trial(self, trial):
        n_heads = trial.suggest_int('n_heads', 1, 2)
        n_hidden_units = trial.suggest_int('n_hidden_units', n_heads, 32, step=n_heads)
        return self.create(
            patch_phi = trial.suggest_categorical('patch_phi', [1, 2, 3, 6]), 
            patch_theta = trial.suggest_categorical('patch_theta', [1, 2, 7]), 
            n_heads = n_heads,
            n_hidden_units = n_hidden_units, 
            n_mlp_units = n_hidden_units, #trial.suggest_int('n_mlp_units', 1, 64), 
            n_ViT = trial.suggest_int('n_ViT', 1, 2), 
            )

    def get_model(self, params):
        return self.create(
            patch_phi = params['patch_phi'], 
            patch_theta = params['patch_theta'], 
            n_heads = params['n_heads'], 
            n_hidden_units = params['n_hidden_units'], 
            n_mlp_units = params['n_mlp_units'], 
            n_ViT = params['n_ViT'], 
            )

    def qmlp(self, x, n_mlp_units):
        x = QDense(
            n_mlp_units, 
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0), 
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0), 
            )(x)
        x = QActivation("quantized_relu(10, 6)")(x)
        x = QDense(
            n_mlp_units, 
            kernel_quantizer=quantized_bits(8, 1, 1, alpha=1.0), 
            bias_quantizer=quantized_bits(8, 3, 1, alpha=1.0), 
            )(x)
        x = QActivation("quantized_relu(10, 6)")(x)
        return x

    def vit_block(self, x, n_hidden_units, n_heads, n_mlp_units):
        # Layer Normalization before attention
        x_norm = LayerNormalization()(x)

        # Quantized Multi-Head Self-Attention
        attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=n_hidden_units // n_heads)(query=x_norm, value=x_norm)
        
        # Residual Connection
        x = tf.keras.layers.Add()([x, attn_output])

        # Layer Normalization before MLP
        x_norm = LayerNormalization()(x)

        # Quantized MLP
        mlp_output = self.qmlp(x_norm, n_mlp_units)

        # Residual Connection
        x = tf.keras.layers.Add()([x, mlp_output])

        return x

    def create(self, patch_phi, patch_theta, n_hidden_units, n_heads, n_mlp_units, n_ViT):
        inputs = Input(shape=self.input_shape, name = 'input')
        x = Reshape((18, 14, 1))(inputs)
        
        # Patch Embedding
        x = PatchEmbedding((patch_phi, patch_theta), n_hidden_units)(x)
        
        # ViT Block(s)
        for i in range(n_ViT):
            x = self.vit_block(x, n_hidden_units, n_heads, n_mlp_units)
        
        # Output
        x = Flatten()(x)
        x = QDense(1, kernel_quantizer=quantized_bits(12, 3, 1, alpha=1.0), use_bias=False)(x)
        x = QActivation("quantized_relu(16, 8)", name='output')(x)

        model = Model(inputs, x, name="ViT")
        return model
