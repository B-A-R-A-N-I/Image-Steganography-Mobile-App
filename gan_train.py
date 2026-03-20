import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from gan_models import build_feature_discriminator, build_image_discriminator
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Create results directory
os.makedirs("results", exist_ok=True)

# Track losses and accuracy
g_losses_epoch, d1_losses_epoch, d2_losses_epoch = [], [], []
msg_accuracies = []

# Load encoder and decoder
encoder = load_model("encoder.h5")
decoder = load_model("decoder.h5")

# Discriminators
G1 = build_feature_discriminator()
G2 = build_image_discriminator()

# Compile Discriminators
bce = BinaryCrossentropy()
G1.compile(optimizer=Adam(1e-4), loss=bce, metrics=["accuracy"])
G2.compile(optimizer=Adam(1e-4), loss=bce, metrics=["accuracy"])

# Load and preprocess CIFAR-10
(x_train, _), (_, _) = cifar10.load_data()

def preprocess_images(images):
    resized = np.array([cv2.resize(img, (64, 64)) for img in images])
    return resized / 255.0

def generate_random_messages(n):
    return (np.random.randint(0, 2, (n, 64, 64, 1)) * 2 - 1).astype(np.float32)

x_train = preprocess_images(x_train)[:10000]
y_train = generate_random_messages(10000)

# Generator = encoder + decoder
def build_generator_output(images, messages):
    stego_features = encoder([images, messages])
    reconstructed_images, extracted_messages = decoder(stego_features)
    return stego_features, reconstructed_images, extracted_messages

# Optimizer and loss
optimizer = Adam(learning_rate=1e-4)
mse = MeanSquaredError()

# Training loop
epochs = 20
batch_size = 32
steps_per_epoch = len(x_train) // batch_size

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    idx = np.random.permutation(len(x_train))
    epoch_g_loss, epoch_d1_loss, epoch_d2_loss = [], [], []
    correct_msgs = 0
    total_msgs = 0

    # tqdm batch loop
    with tqdm(total=steps_per_epoch, desc="Progress", ncols=120) as pbar:
        for i in range(0, len(x_train), batch_size):
            batch_idx = idx[i:i + batch_size]
            images = x_train[batch_idx]
            messages = y_train[batch_idx]

            # Generator forward pass
            stego_features, reconstructed_images, extracted_messages = build_generator_output(images, messages)

            # Labels
            real = np.ones((len(images), 1))
            fake = np.zeros((len(images), 1))

            # Train discriminators
            g1_loss_real = G1.train_on_batch(encoder([images, messages]), real)
            g1_loss_fake = G1.train_on_batch(stego_features, fake)
            g2_loss_real = G2.train_on_batch(images, real)
            g2_loss_fake = G2.train_on_batch(reconstructed_images, fake)

            d_loss_g1 = 0.5 * np.add(g1_loss_real, g1_loss_fake)
            d_loss_g2 = 0.5 * np.add(g2_loss_real, g2_loss_fake)

            # Train generator
            with tf.GradientTape() as tape:
                features = encoder([images, messages])
                recon_images, extracted_msgs = decoder(features)

                validity_g1 = G1(features)
                validity_g2 = G2(recon_images)

                g1_adv_loss = bce(tf.ones_like(validity_g1), validity_g1)
                g2_adv_loss = bce(tf.ones_like(validity_g2), validity_g2)

                image_loss = mse(images, recon_images)
                message_loss = mse(messages, extracted_msgs)

                g_loss = image_loss + 50 * message_loss + 0.1 * (g1_adv_loss + g2_adv_loss)

            grads = tape.gradient(g_loss, encoder.trainable_variables + decoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables))

            # Accuracy
            predicted = tf.math.sign(extracted_msgs).numpy()
            correct = np.sum(predicted == messages)
            correct_msgs += correct
            total_msgs += np.prod(messages.shape)

            # Record metrics
            epoch_g_loss.append(g_loss.numpy())
            epoch_d1_loss.append(d_loss_g1[0])
            epoch_d2_loss.append(d_loss_g2[0])

            avg_g = np.mean(epoch_g_loss)
            avg_d1 = np.mean(epoch_d1_loss)
            avg_d2 = np.mean(epoch_d2_loss)
            avg_acc = correct_msgs / total_msgs

            # Update tqdm bar
            pbar.set_postfix({
                "g_loss": f"{avg_g:.4f}",
                "d1_loss": f"{avg_d1:.4f}",
                "d2_loss": f"{avg_d2:.4f}",
                "accuracy": f"{avg_acc:.4f}"
            })
            pbar.update(1)

    # Store epoch-level metrics
    g_losses_epoch.append(avg_g)
    d1_losses_epoch.append(avg_d1)
    d2_losses_epoch.append(avg_d2)
    msg_accuracies.append(avg_acc)

    print(f"✔️ Epoch {epoch + 1} complete. Accuracy: {avg_acc:.4f}")

# Save models
encoder.save("models/encoder_gan.h5")
decoder.save("models/decoder_gan.h5")
G1.save("models/discriminator_feature.h5")
G2.save("models/discriminator_image.h5")
print("Models saved.")

# Plot loss and accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(g_losses_epoch, label='Generator Loss')
plt.plot(d1_losses_epoch, label='Discriminator G1 Loss')
plt.plot(d2_losses_epoch, label='Discriminator G2 Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(msg_accuracies, label='Message Accuracy', color='green')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Message Extraction Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("results/gan_epoch_loss_accuracy.png")
plt.show()