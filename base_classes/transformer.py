# Much of the implementation came from the Tensorflow Transformer tutorial:
# https://www.tensorflow.org/text/tutorials/transformer

# ########################################################################
# # Setup
# ########################################################################
import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
import math

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_text as tf_text
import requests

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

# ########################################################################
# # Dataset
# ########################################################################

# Sequence Settings
MAX_REPORT_LENGTH = 1500 # In number of tokens
BUFFER_SIZE = 20000
BATCH_SIZE = 8
start_token = 2
end_token = 3
# Hyperparameters
num_layers = 4 # 6
d_model = 128 # 128
dff = 256 # 512
num_heads = 4 # 8
dropout_rate = 0.1
# Tokenizer
tokenizer_vocab_filepath = "ufo_vocab.txt"
tokenizer = tf_text.BertTokenizer(tokenizer_vocab_filepath, lower_case=True) #  token_out_type=tf.string,
# tokenizer = tf_text.UnicodeScriptTokenizer()


# ########################################################################
# # Text tokenization & detokenization
# ########################################################################

def download_vocabulary():
    url = "https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/test_data/test_wp_en_vocab.txt?raw=true"
    r = requests.get(url)
    open(tokenizer_vocab_filepath, 'wb').write(r.content)


def tokenize(to_tok):
    tokenized = tokenizer.tokenize(to_tok).merge_dims(1, 2)
    start_mask = tf.repeat([[start_token]], repeats=[tokenized.get_shape()[0]], axis=0)
    end_mask = tf.repeat([[end_token]], repeats=[tokenized.get_shape()[0]], axis=0)
    return tf.concat([start_mask, tf.cast(tokenized, tf.int32), end_mask], 1)

def detokenize(to_detok):
    return tokenizer.detokenize(to_detok)

def pure_tokenize(to_tok):
    return tokenizer.tokenize(to_tok)#.merge_dims(1, 2)

def tokenize_pairs(inp, tar):
    inp = tokenize(inp)
    inp = inp.to_tensor()
    tar = tokenize(tar)
    tar = tar.to_tensor()
    return inp, tar




def create_partition_batches(ds, ds_size, train_split=0.7, val_split=0.2, test_split=0.1):
    ds = ds.shuffle(BUFFER_SIZE)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    test_size = int(test_split * ds_size)
    
    train_ds = ds.take(train_size).batch(BATCH_SIZE)
    val_ds = ds.skip(train_size).take(val_size).batch(BATCH_SIZE)
    test_ds = ds.skip(train_size).skip(val_size).batch(BATCH_SIZE)
    # test_ds = ds.skip(train_size).skip(val_size).take(test_size).batch(BATCH_SIZE)

    return train_ds, val_ds, test_ds


# ########################################################################
# # Positional encoding
# ########################################################################


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# ########################################################################
# # Masking
# ########################################################################


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# ########################################################################
# # Scaled dot product attention
# ########################################################################


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

# ########################################################################
# # Multi-head attention
# ########################################################################


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)  # query weights
        self.wk = tf.keras.layers.Dense(d_model)  # key weights
        self.wv = tf.keras.layers.Dense(d_model)  # value weights

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth)) # does the split
        return tf.transpose(x, perm=[0, 2, 1, 3])                       # does the transpose

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights


# ########################################################################
# # Point wise feed forward network
# ########################################################################


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(dff, activation='relu'),  # dff == # hidden layer nodes
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


# ########################################################################
# # Encoder layer
# ########################################################################


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # Send input through multi-head attention layer
        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        # Send that through the FFN
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


# ########################################################################
# # Decoder layer
# ########################################################################


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


# ########################################################################
# # Encoder
# ########################################################################


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 pe_input, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(pe_input, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


# ########################################################################
# # Decoder
# ########################################################################


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 pe_target, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            pe_target, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

# ########################################################################
# # Create the Transformer
# ########################################################################


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, is_training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(
            inp, tar)

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, is_training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, is_training, look_ahead_mask, dec_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask


# ########################################################################
# # Optimizer
# ########################################################################
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

# ########################################################################
# # Loss and metrics
# ########################################################################
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


# ########################################################################
# # Training and checkpointing
# ########################################################################
def build_transformer():
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=7816,
        target_vocab_size=7816,
        pe_input=10000,
        pe_target=6000,
        rate=dropout_rate)

    checkpoint_path = "./training_checkpoints_Transformer/full"

    ckpt = tf.train.Checkpoint(transformer=transformer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    
    return transformer

def load_transformer():
    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=7816,
        target_vocab_size=7816,
        pe_input=10000,
        pe_target=6000,
        rate=dropout_rate)

    checkpoint_path = "./models/ufo_rep_transformer"

    ckpt = tf.train.Checkpoint(transformer=transformer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Transformer Model restored!')
    
    return transformer

# def save_transformer():
#     transformer = build_transformer()

#     print("Running test prediction")
#     output_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
#     output_array = output_array.write(0, tf.constant([start_token]))
#     encoder_input = tf.constant([start_token, end_token])[tf.newaxis]
#     output = tf.transpose(output_array.stack())
#     predictions, _ = transformer([encoder_input, output], is_training=False)

#     checkpoint_path = "./models/ufo_rep_transformer"
#     ckpt = tf.train.Checkpoint(transformer=transformer)
#     ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
#     ckpt_manager.save()
#     print("Transformer Model saved!")


# transformer = build_transformer()
transformer = load_transformer()












# checkpoint_path = "./training_checkpoints_Transformer/full"

# ckpt = tf.train.Checkpoint(transformer=transformer,
#                            optimizer=optimizer)

# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)

# # if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest checkpoint restored!!')



# # The @tf.function trace-compiles train_step into a TF graph for faster
# # execution. The function specializes to the precise shape of the argument
# # tensors. To avoid re-tracing due to the variable sequence lengths or variable
# # batch sizes (the last batch is smaller), use input_signature to specify
# # more generic shapes.

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp],
                                     is_training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(tar_real, predictions))

def continue_training(train_batches, num_epochs=5):
    tf.print("Continuing training")
    EPOCHS = num_epochs

    # Attempts to load prior checkpoint
    checkpoint_path = "./training_checkpoints_Transformer/full"
    ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)
    # # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(tf.cast(inp, tf.int64), tf.cast(tar, tf.int64))

            if batch % 50 == 0:
                tf.print(
                    f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        if (epoch + 1) % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            tf.print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        tf.print(
            f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        tf.print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')




# output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
# output_array = output_array.write(0, tokenize_sequence(tf.constant([start_token])))
# encoder_input = tokenize_sequence(tf.constant([start_token]))[tf.newaxis]
# seq_len = tf.size(encoder_input)
# for i in tf.range(10):
#     # print(seq_len)
#     # print(GENOME_CUTOFF_LEN - seq_len.numpy())
#     output = tf.transpose(output_array.stack())
#     # paddings = tf.constant([[0,0], [0, GENOME_CUTOFF_SIZE - tf.size(encoder_input).numpy()]])
#     # padded_input = tf.pad(encoder_input, paddings, mode='CONSTANT')
#     print("encoder input: ", encoder_input)
#     # print("padded input: ", padded_input)
#     # predictions, _ = transformer([encoder_input, output], is_training=False)
#     predictions, _ = transformer([encoder_input, output], is_training=False)

#     # select the last token from the seq_len dimension
#     predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

#     print("predictions: ", predictions)
#     print("prediction probabilities: ", tf.nn.softmax(predictions))
#     print("max probability: ", tf.reduce_max(tf.nn.softmax(predictions)))

#     predicted_id = tf.argmax(predictions, axis=-1)

#     # concatentate the predicted_id to the output which is given to the decoder
#     # as its input.
#     print("predicted id: ", predicted_id)
#     seq_len += 1
#     # encoder_input = tf.stack([tf.cast(encoder_input, tf.int32), tf.cast(predicted_id, tf.int32)])
#     # encoder_input = tf.reshape(tf.concat([tf.cast(encoder_input, tf.int32), tf.cast(predicted_id, tf.int32)], 1), [1, seq_len])
#     output_array = output_array.write(i+1, predicted_id[0])
#     print(output_array)

#     # if predicted_id == 6:
#     #     break

# output = tf.transpose(output_array.stack())
# print(output)
# print(detokenize_sequence(output))
