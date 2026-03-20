from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Flatten, Dense

def build_feature_discriminator(input_shape=(64, 64, 8)):  # <-- was 16
    inp = Input(shape=input_shape, name="Feature_Input")

    x = Conv2D(32, (3, 3), strides=2, padding='same')(inp)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid', name="Feature_Validity")(x)

    return Model(inp, output, name="FeatureDiscriminator")


def build_image_discriminator(input_shape=(64, 64, 3)):
    inp = Input(shape=input_shape, name="Image_Input")

    x = Conv2D(32, (3, 3), strides=2, padding='same')(inp)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid', name="Image_Validity")(x)

    return Model(inp, output, name="ImageDiscriminator")
