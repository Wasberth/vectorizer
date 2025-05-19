import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Simple Vision Transformer (ViT)-like encoder for patches
class SimpleViTEncoder(tf.keras.Model):
    def __init__(self, patch_size=4, embed_dim=64, num_patches=64):
        super().__init__()
        self.patch_proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')
        self.flatten = layers.Reshape((num_patches, embed_dim))
        self.encoder_layers = [
            layers.LayerNormalization(),
            layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim),
            layers.Dense(embed_dim, activation='relu'),
        ]

    def call(self, x):
        x = self.patch_proj(x)
        x = self.flatten(x)
        for layer in self.encoder_layers:
            if isinstance(layer, layers.MultiHeadAttention):
                x = layer(x, x)
            else:
                x = layer(x)
        return x

# Autoregressive Transformer Decoder for coordinate generation
class AutoregressiveDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim=64, num_heads=4, mlp_dim=128, max_len=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.coord_embedding = layers.Dense(embed_dim)
        self.pos_embedding = self.add_weight(
            shape=(max_len + 1, embed_dim),
            initializer="random_normal",
            trainable=True,
            name="pos_embedding"
        )
        self.decoder_layers = [
            {
                "self_attn": layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim),
                "cross_attn": layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim),
                "norm1": layers.LayerNormalization(),
                "norm2": layers.LayerNormalization(),
                "ffn": models.Sequential([
                    layers.Dense(mlp_dim, activation='relu'),
                    layers.Dense(embed_dim)
                ])
            }
            for _ in range(2)
        ]
        self.output_head = layers.Dense(2)  # [x, y]

    def call(self, encoder_output, training=False):
        batch_size = tf.shape(encoder_output)[0]
        coords = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        prev_coord = tf.zeros((batch_size, 1, 2))  # [B, 1, 2]
        embedded_seq = self.coord_embedding(prev_coord)

        for t in range(self.max_len):
            x = embedded_seq + self.pos_embedding[:tf.shape(embedded_seq)[1]]

            for layer in self.decoder_layers:
                attn_mask = tf.linalg.band_part(tf.ones((tf.shape(x)[1], tf.shape(x)[1])), -1, 0)
                x = layer["norm1"](x + layer["self_attn"](x, x, attention_mask=attn_mask))
                x = layer["norm2"](x + layer["cross_attn"](x, encoder_output))
                x = x + layer["ffn"](x)

            next_coord = self.output_head(x[:, -1:, :])
            coords = coords.write(t, next_coord)

            if not training:
                if tf.reduce_all(tf.equal(tf.round(next_coord), tf.constant([[[-1.0, -1.0]]], dtype=tf.float32))):
                    break

            embedded_seq = tf.concat([embedded_seq, self.coord_embedding(next_coord)], axis=1)

        return tf.transpose(tf.squeeze(coords.stack(), axis=2), [1, 0, 2])

# Integrate Encoder + Decoder into full model
class VisionTransformerCoordinateModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = SimpleViTEncoder(patch_size=4, embed_dim=64, num_patches=64)
        self.decoder = AutoregressiveDecoder(embed_dim=64, num_heads=4, mlp_dim=128, max_len=10)

    def call(self, x, training=False):
        encoded = self.encoder(x)
        coords = self.decoder(encoded)
        return coords

def generate_synthetic_data(num_samples=1000, image_size=32, max_points=5):
    images = []
    coord_seqs = []

    for _ in range(num_samples):
        num_points = np.random.randint(1, max_points + 1)
        coords = np.random.rand(num_points, 2)  # valores entre [0, 1]

        # Crear imagen negra y dibujar puntos blancos
        img = np.zeros((image_size, image_size, 3), dtype=np.float32)
        for x, y in coords:
            xi = int(x * (image_size - 1))
            yi = int(y * (image_size - 1))
            img[yi, xi] = [1.0, 1.0, 1.0]  # punto blanco

        # Padding de coordenadas
        pad_len = 10 - num_points
        padded_coords = np.vstack([coords, np.full((pad_len, 2), -1.0)])

        images.append(img)
        coord_seqs.append(padded_coords)

    return np.array(images, dtype=np.float32), np.array(coord_seqs, dtype=np.float32)

def loss_fn(y_true, y_pred):
    """
    y_true: [B, T, 2], y_pred: [B, T, 2]
    Ignora coordenadas con valor [-1, -1]
    """
    mask = tf.reduce_all(y_true != -1.0, axis=-1)
    loss = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)
    masked_loss = tf.where(mask, loss, tf.zeros_like(loss))
    return tf.reduce_mean(masked_loss)

def inferir_y_visualizar(modelo, imagen, image_size=(224, 224), stop_token=[-1.0, -1.0]):
    """
    Ejecuta inferencia sobre una imagen y dibuja las coordenadas predichas.

    Args:
        modelo: modelo cargado (tf.keras.Model)
        imagen: imagen como np.array (H, W, 3)
        image_size: tamaño al que se debe redimensionar la imagen
        stop_token: valor especial para detener la secuencia (por defecto: [-1, -1])
    """
    # Preprocesar imagen
    img_resized = tf.image.resize(imagen, image_size) / 255.0
    img_batch = tf.expand_dims(img_resized, axis=0)  # [1, H, W, 3]

    # Inferencia
    coords_pred = modelo(img_batch, training=False).numpy()[0]  # [T, 2]

    # Filtrar coordenadas válidas
    coords = []
    for coord in coords_pred:
        if np.allclose(coord, stop_token, atol=1e-3):
            break
        coords.append(coord)
    coords = np.array(coords)

    # Visualización
    plt.figure(figsize=(5, 5))
    plt.imshow(imagen.astype(np.uint8))
    if len(coords) > 0:
        h, w = imagen.shape[:2]
        xs = coords[:, 0] * w
        ys = coords[:, 1] * h
        plt.plot(xs, ys, 'ro-')
    plt.axis('off')
    plt.title('Coordenadas predichas')
    plt.show()


if __name__ == '__main__':
    model = VisionTransformerCoordinateModel()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # Dataset
    images, coords = generate_synthetic_data(1000)
    train_ds = tf.data.Dataset.from_tensor_slices((images, coords)).batch(32).shuffle(1000)

    # Loop de entrenamiento
    EPOCHS = 5

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, (x_batch, y_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=True)
                loss = loss_fn(y_batch, y_pred)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            total_loss += loss.numpy()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / (step + 1):.4f}")
    
    model.save('models/transformer.keras')

    inferir_y_visualizar(model, images[0], (32, 32))