# coding=utf-8
# Copyright 2023 Thang V Pham
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import re

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

DATA_DIR = os.environ['HOME'] + '/data'

#region ms_data

def min_max_scale(x, min, max):
    new_x = 1.0 * (x - min) / (max - min)
    return new_x

def min_max_scale_rev(x, min, max):
    old_x = x * (max - min) + min
    return old_x

class data_phospho():

    @classmethod
    def load_training_transformer(cls):
        
        x_train = np.load(DATA_DIR + '/HumanPhosphoproteomeDB_rt_x_train.npy')
        y_train = np.load(DATA_DIR + '/HumanPhosphoproteomeDB_rt_y_train.npy')

        x_val = np.load(DATA_DIR + '/HumanPhosphoproteomeDB_rt_x_val.npy')
        y_val = np.load(DATA_DIR + '/HumanPhosphoproteomeDB_rt_y_val.npy')

        return (x_train, y_train), (x_val, y_val)

    @classmethod
    def load_testing_transformer(cls):
        x_test = np.load(DATA_DIR + '/HumanPhosphoproteomeDB_rt_x_test.npy')
        y_test = np.load(DATA_DIR + '/HumanPhosphoproteomeDB_rt_y_test.npy')

        return (x_test, y_test)

class data_autort():

    @classmethod
    def load_training_transformer(cls, max_sequence_length = 48):
        
        x_train = np.load(DATA_DIR + '/PXD006109_rt_x_train.npy')
        y_train = np.load(DATA_DIR + '/PXD006109_rt_y_train.npy')

        x_val = np.load(DATA_DIR + '/PXD006109_rt_x_val.npy')
        y_val = np.load(DATA_DIR + '/PXD006109_rt_y_val.npy')

        return (x_train, y_train), (x_val, y_val)

    @classmethod
    def load_testing_transformer(cls, max_sequence_length = 48):
        x_test = np.load(DATA_DIR + '/PXD006109_rt_x_test.npy')
        y_test = np.load(DATA_DIR + '/PXD006109_rt_y_test.npy')

        return (x_test, y_test)


class data_deepdia():

    @classmethod
    def load_training_transformer(cls, max_sequence_length = 50):

        x_train = np.load(DATA_DIR + '/PXD005573_rt_x_train.npy')
        y_train = np.load(DATA_DIR + '/PXD005573_rt_y_train.npy')

        x_val = np.load(DATA_DIR + '/PXD005573_rt_x_val.npy')
        y_val = np.load(DATA_DIR + '/PXD005573_rt_y_val.npy')

        return (x_train, y_train), (x_val, y_val)

    @classmethod
    def load_testing_transformer(cls, max_sequence_length = 50):

        x_test = np.load(DATA_DIR + '/PXD005573_rt_x_test.npy')
        y_test = np.load(DATA_DIR + '/PXD005573_rt_y_test.npy')

        return (x_test, y_test)

class data_prosit():
    @classmethod
    def load_training(cls, max_sequence_length = 30):

        x_train = np.load(DATA_DIR + '/prosit_rt_updated/X_train.npy')
        y_train = np.load(DATA_DIR + '/prosit_rt_updated/Y_train.npy')

        x_val = np.load(DATA_DIR + '/prosit_rt_updated/X_validation.npy')
        y_val = np.load(DATA_DIR + '/prosit_rt_updated/Y_validation.npy')

        return (x_train, y_train), (x_val, y_val)


    @classmethod
    def load_testing(cls, max_sequence_length = 30):

        x_test = np.load(DATA_DIR + '/prosit_rt_updated/X_holdout.npy')
        y_test = np.load(DATA_DIR + '/prosit_rt_updated/Y_holdout.npy')
        
        return (x_test, y_test)

class data_generics():
    
    @classmethod
    def sequence_to_integer(cls, sequences, max_sequence_length):

        Prosit_ALPHABET = {
            'A': 1,
            'C': 2,
            'D': 3,
            'E': 4,
            'F': 5,
            'G': 6,
            'H': 7,
            'I': 8,
            'K': 9,
            'L': 10,
            'M': 11,
            'N': 12,
            'P': 13,
            'Q': 14,
            'R': 15,
            'S': 16,
            'T': 17,
            'V': 18,
            'W': 19,
            'Y': 20,
            'o': 21,
        }

        array = np.zeros([len(sequences), max_sequence_length], dtype=int)
        for i, sequence in enumerate(sequences):
            for j, s in enumerate(re.sub('M\(ox\)', 'o', sequence)):
                array[i, j] = Prosit_ALPHABET[s]
        return array

    @classmethod
    def load_deepdia(cls, filename, seq_header = 'sequence', rt_header = 'rt'):

        min_sequence_length = 7
        max_sequence_length = 50

        d = pd.read_csv(filename)

        if seq_header not in d:
            print('No column in the data: ' + seq_header)
            exit(0)

        has_rt = rt_header in d

        print(d.shape[0], ' peptides')

        selected_peptides = {}
        for index, row in d.iterrows():
            s = row[seq_header][1:-1] # remove _xxx_

            if s.find('(') < 0 and (min_sequence_length <= len(s) <= max_sequence_length):
                if has_rt :
                    selected_peptides[s] = row['rt']
                else :
                    selected_peptides[s] = 0

        print(len(selected_peptides), ' peptides selected')

        df = pd.DataFrame.from_dict(selected_peptides, orient = 'index', columns = ['rt'])
        x = cls.sequence_to_integer(df.index.values, max_sequence_length)

        return (x, df['rt'].to_numpy())


    @classmethod
    def load_prosit(cls, filename, seq_header = 'sequence', rt_header = 'rt'):

        min_sequence_length = 7
        max_sequence_length = 30

        d = pd.read_csv(filename)

        if seq_header not in d:
            print('No column in the data: ' + seq_header)
            exit(0)

        has_rt = rt_header in d

        print(d.shape[0], ' peptides')

        selected_peptides = {}
        for index, row in d.iterrows():
            s = row[seq_header][1:-1] # remove _xxx_
            s = re.sub('M\(ox\)', 'o', s)
            if s.find('(') < 0 and (min_sequence_length <= len(s) <= max_sequence_length):
                if has_rt :
                    selected_peptides[s] = row['rt']
                else :
                    selected_peptides[s] = 0


        print(len(selected_peptides), ' peptides selected')

        df = pd.DataFrame.from_dict(selected_peptides, orient = 'index', columns = ['rt'])
        x = cls.sequence_to_integer(df.index.values, max_sequence_length)

        return (x, df['rt'].to_numpy())

    @classmethod
    def load_autort(cls, filename, seq_header = 'sequence', rt_header = 'rt'):

        min_sequence_length = 7
        max_sequence_length = 48

        d = pd.read_csv(filename)

        if seq_header not in d:
            print('No column in the data: ' + seq_header)
            exit(0)

        has_rt = rt_header in d


        print(d.shape[0], ' peptides')

        selected_peptides = {}
        for index, row in d.iterrows():
            s = row[seq_header][1:-1] # remove _xxx_
            s = re.sub('M\(ox\)', 'o', s)
            if s.find('(') < 0 and (min_sequence_length <= len(s) <= max_sequence_length):
                if has_rt :
                    selected_peptides[s] = row[rt_header]
                else :
                    selected_peptides[s] = 0


        print(len(selected_peptides), ' peptides selected')

        df = pd.DataFrame.from_dict(selected_peptides, orient = 'index', columns = ['rt'])
        x = cls.sequence_to_integer(df.index.values, max_sequence_length)

        return (x, df['rt'].to_numpy())

    @classmethod
    def integer_to_sequence(cls, X):
        int2seq = '_ACDEFGHIKLMNPQRSTVWYo'
        int2seqf = lambda x : ''.join([int2seq[c] for c in x if c > 0])
        return ([int2seqf(x) for x in X])

    @classmethod
    def sequence_to_integer_phospho(cls, sequences, max_sequence_length = 60):

        deepphospho_ALPHABET = {
            "A": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "K": 9,
            "L": 10,
            "M": 11,
            "N": 12,
            "P": 13,
            "Q": 14,
            "R": 15,
            "S": 16,
            "T": 17,
            "V": 18,
            "W": 19,
            "Y": 20,
            "1": 21,
            "2": 22,
            "3": 23,
            "4": 24,
            "*": 25
        }

        array = np.zeros([len(sequences), max_sequence_length], dtype=int)
        for i, sequence in enumerate(sequences):
            for j, s in enumerate(re.sub('@', '', sequence)):
                array[i, j] = deepphospho_ALPHABET[s]
        return array

    @classmethod
    def load_phospho(cls, filename, seq_header = 'IntPep', rt_header = 'iRT'):

        min_sequence_length = 6
        max_sequence_length = 60

        d = pd.read_csv(filename)

        if seq_header not in d:
            print('No column in the data: ' + seq_header)
            exit(0)

        has_rt = rt_header in d


        print(d.shape[0], ' peptides')

        selected_peptides = {}
        for index, row in d.iterrows():
            s = row[seq_header]
            s = re.sub('@', '', s)
            if s.find('(') < 0 and (min_sequence_length <= len(s) <= max_sequence_length):
                if has_rt :
                    selected_peptides[s] = row[rt_header]
                else :
                    selected_peptides[s] = 0
            else:
                print(s)


        print(len(selected_peptides), ' peptides selected')

        df = pd.DataFrame.from_dict(selected_peptides, orient = 'index', columns = ['rt'])

        x = cls.sequence_to_integer_phospho(df.index.values, max_sequence_length)

        return (x, df['rt'].to_numpy())

    @classmethod
    def integer_to_sequence_phospho(cls, X):
        int2seq = '_ACDEFGHIKLMNPQRSTVWY1234*'
        int2seqf = lambda x : ''.join([int2seq[c] for c in x if c > 0])
        return ([int2seqf(x) for x in X])

#endregion


#region Transformer

# Code in the Transformer region is based on https://www.tensorflow.org/text/tutorials/transformer

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


# Position
def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# Masking
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# Scaled dot product attention
def scaled_dot_product_attention(q, k, v, mask):
    '''
    Calculate the attention weights.

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
    '''

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += mask * -1e9

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1
    )  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# Multi-head attention
class multi_head_attention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(multi_head_attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        '''
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        '''
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model)
        )  # (batch_size, seq_len_q, d_model)


        # BERT dense layer is here. see class TFBertSelfOutput(tf.keras.layers.Layer):
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# Point wise feed forward network
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation='gelu'),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


# Encoder and decoder
class encoder_layer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(encoder_layer, self).__init__()

        self.mha = multi_head_attention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        out1 = self.layernorm1(x)  # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(out1, out1, out1, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = x + attn_output  # (batch_size, input_seq_len, d_model)

        out2 = self.layernorm2(out1)
        ffn_output = self.ffn(out2)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = out1 + ffn_output

        return out2


# Encoder
class encoder_block(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        maximum_position_encoding,
        rate = 0.1,
    ):
        super(encoder_block, self).__init__(name = 'transformer')        

        self.d_model = d_model
        self.num_layers = num_layers        
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = (input_vocab_size,)
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate        

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        #self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.pos_embedding = tf.keras.layers.Embedding(maximum_position_encoding, d_model)

        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.enc_layers = [
            encoder_layer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):

        seq_len = tf.shape(x)[1]

        enc_padding_mask = create_padding_mask(x)

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        
        #x += self.pos_encoding[:, :seq_len, :]
        indices = tf.range(self.maximum_position_encoding)
        x += self.pos_embedding(indices)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, enc_padding_mask)

        x = self.layernorm(x)
        
        return x  # (batch_size, input_seq_len, d_model)

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                'num_layers': self.num_layers,
                'd_model': self.d_model,
                'num_head': self.num_heads,
                'dff': self.dff,
                'input_vocab_size': self.input_vocab_size,
                'maximum_position_encoding': self.maximum_position_encoding,
                'rate': self.rate,
            }
        )
        return config

class custom_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps = 4000):
        super(custom_schedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(tf.cast(step, dtype = tf.float32))
        arg2 = tf.cast(step, dtype = tf.float32) * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

#endregion


#region Model training
def build_model(num_layers, d_model, num_heads, d_ff, dropout_rate, vocab_size, max_len):

    coded_input = tf.keras.layers.Input(shape=(max_len,), name = 'input')

    encoder = encoder_block(num_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout_rate)

    enc_output = encoder(coded_input)  # (batch_size, inp_seq_len, d_model)

    net = enc_output[:, 0, :]

    net = tf.keras.layers.Dense(512, activation = 'gelu', name = 'predict_1')(net)
    net = tf.keras.layers.Dropout(dropout_rate, name = 'predict_dropout_1')(net)
    net = tf.keras.layers.Dense(512, activation = 'gelu', name = 'predict_2')(net)
    net = tf.keras.layers.Dropout(dropout_rate, name = 'predict_dropout_2')(net)
    #net = tf.keras.layers.Dense(1, activation = 'linear', name = 'output')(net)
    net = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'output')(net)

    m = tf.keras.Model(coded_input, net, name = 'rt_transformer')
    m.layers[2]._name = 'CLS_token'

    return m

#endregion



def main():

    if len(sys.argv) == 1:
        print('python rt.py [tune, train, predict]')
        sys.exit(0)
    else:

        mode = sys.argv[1]

        if mode == 'train':
            '''
            Example:
            python ./rt.py train -data prosit -logs logs-train-prosit -epochs 5000 -n_layers 8 -n_heads 8 -dropout 0.1 -batch_size 128 -d_model 256 -d_ff 1024
            '''
            parser = argparse.ArgumentParser()
            parser.add_argument('-data', '--data', default = 'prosit', type = str, help = 'Data for training, default prosit.')
            parser.add_argument('-batch_size', '--batch_size', default = 1024, type = int, help = 'Batch size for training, default 1024.')
            parser.add_argument('-d_model', '--d_model', default = 512, type = int, help = 'd_model, default 512.')
            parser.add_argument('-n_layers', '--n_layers', default = 10, type = int, help = 'n_layers, default 10.')
            parser.add_argument('-n_heads', '--n_heads', default = 8, type = int, help = 'n_heads, default 8.')
            parser.add_argument('-d_ff', '--d_ff', default = 512, type = int, help = 'd_ff, default 512.')
            parser.add_argument('-dropout', '--dropout', default = 0.1, type = float, help = 'dropout, default 0.1.')
            parser.add_argument('-epochs', '--epochs', default = 2000, type = int, help = 'Number of epochs, default 2000.')
            parser.add_argument('--gpu', default=True, action=argparse.BooleanOptionalAction)
            parser.add_argument('-logs', '--logs', default = 'logs', type = str, help = 'Directory for logging')

            args = parser.parse_args(sys.argv[2:len(sys.argv)])

            print('data =', args.data)
            print('batch_size =', args.batch_size)
            print('d_model =', args.d_model)
            print('d_ff =', args.d_ff)
            print('n_layers =', args.n_layers)
            print('n_heads =', args.n_heads)
            print('dropout =', args.dropout)
            print('epochs =', args.epochs)

            if not args.gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            if args.data == 'autort':
                (x_train, y_train), (x_val, y_val) = data_autort.load_training_transformer()

                min_val = 0.0
                max_val = 101.33

            elif args.data == 'prosit':
                (x_train, y_train), (x_val, y_val) = data_prosit.load_training()
                
                min_val = min(y_train.min(), y_val.min()) - 0.01
                max_val = max(y_train.max(), y_val.max()) + 0.01
                
                print("(min, max) = ", (min_val, max_val))

                y_train = min_max_scale(y_train, min = min_val, max = max_val)
                y_val = min_max_scale(y_val, min = min_val, max = max_val)

            elif args.data == 'deepdia':
                (x_train, y_train), (x_val, y_val) = data_deepdia.load_training_transformer()
                
                min_val = min(y_train.min(), y_val.min()) - 0.01
                max_val = max(y_train.max(), y_val.max()) + 0.01
                
                print("(min, max) = ", (min_val, max_val))

                y_train = min_max_scale(y_train, min = min_val, max = max_val)
                y_val = min_max_scale(y_val, min = min_val, max = max_val)

            elif args.data == 'phospho':
                (x_train, y_train), (x_val, y_val) = data_phospho.load_training_transformer()

                min_val = 0.0
                max_val = 1.0

            else:
                print('Unknown data')
                exit(0)

            CLS = x_train.max() + 1
            x_train = np.concatenate((np.full((x_train.shape[0], 1), CLS), x_train), axis = 1)
            x_val = np.concatenate((np.full((x_val.shape[0], 1), CLS), x_val), axis = 1)

            print(len(x_train), 'Training sequences')
            print(len(x_val), 'Validation sequences')

            print('CLS =', CLS)

            cp_path = './' + args.data + '-b' + str(args.batch_size) + '-dm' + str(args.d_model) + '-df' + str(args.d_ff) + '-nl' + str(args.n_layers) + '-nh' + str(args.n_heads) + '-dr' + str(args.dropout) + '-ep' + str(args.epochs) + '/'

            print(cp_path)

            from pathlib import Path
            Path(cp_path).mkdir(parents=True, exist_ok=True)

            pd.DataFrame({'parameter': ['data', 'batch_size', 'd_model', 'd_ff', 'n_layers', 'n_heads', 'dropout', 
                                        'epochs', 'vocab_size', 'max_length', 'min_val', 'max_val'],
                          'value': [args.data, str(args.batch_size), str(args.d_model), str(args.d_ff), str(args.n_layers), str(args.n_heads), str(args.dropout), 
                                    str(args.epochs), str(CLS + 1), str(x_train.shape[1]), str(min_val), str(max_val)]}).to_csv(cp_path + 'parameters.txt', sep= '\t', index = False)


            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = cp_path + 'weights.{epoch:04d}.hdf5',
                                                            save_best_only = True,
                                                            #save_best_only = False,
                                                            save_weights_only = True,
                                                            verbose = 1)

            csv_logger = tf.keras.callbacks.CSVLogger(cp_path + 'model_history_log.csv', append = False)

            tf.random.set_seed(1)
            np.random.seed(1)

            model = build_model(
                num_layers = args.n_layers,
                d_model = args.d_model,
                num_heads = args.n_heads,
                d_ff = args.d_ff,
                dropout_rate = args.dropout,
                vocab_size = CLS + 1,           # number of aminoacids
                max_len = x_train.shape[1]      # maximal peptide length
            )


            np.random.seed(1)
            tf.random.set_seed(1)

            learning_rate = custom_schedule(args.d_model)
            optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9)

            model.compile(optimizer = optimizer, loss = tf.keras.losses.MeanAbsoluteError())

            model.summary()

            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.logs, histogram_freq=1)
            earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 5000, verbose=1)
            cb = [checkpoint, csv_logger, tensorboard_callback, earlystopper]
            
            training_history = model.fit(
                            x_train,
                            y_train,
                            batch_size = args.batch_size,
                            epochs = args.epochs,
                            validation_data = (x_val, y_val),
                            callbacks = cb)

            print('Min test loss: ', np.min(training_history.history['loss']))

        elif mode == 'tune':
            # Tuning using Tensorboard https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams

            parser = argparse.ArgumentParser()
            parser.add_argument('-data', '--data', default = 'prosit', type = str, help = 'Data for training, default prosit.')
            parser.add_argument('-batch_size', '--batch_size', nargs = '+', default = 1024, type = int, help = 'Batch size for training, default 1024.')
            parser.add_argument('-d_model', '--d_model', nargs = '+', default = 512, type = int, help = 'd_model, default 512.')
            parser.add_argument('-n_layers', '--n_layers', nargs = '+', default = 10, type = int, help = 'n_layers, default 10.')
            parser.add_argument('-n_heads', '--n_heads', nargs = '+', default = 8, type = int, help = 'n_heads, default 8.')
            parser.add_argument('-d_ff', '--d_ff', nargs = '+', default = 512, type = int, help = 'd_ff, default 512.')
            parser.add_argument('-dropout', '--dropout', nargs = '+', default = 0.1, type = float, help = 'dropout, default 0.1.')
            parser.add_argument('-epochs', '--epochs', default = 200, type = int, help = 'Number of epochs, default 200.')
            parser.add_argument('--gpu', default=True, action=argparse.BooleanOptionalAction)
            parser.add_argument('-logs', '--logs', default = 'logs-random-search', type = str, help = 'Directory for logging')
            parser.add_argument('-seed', '--seed', default = 0, type = int, help = 'Random seed')
            parser.add_argument('-n_random_samples', '--n_random_samples', default = 200, type = int, help = 'Number of random samples')
            parser.add_argument('-begin', '--begin', default = 0, type = int, help = 'begin')
            parser.add_argument('-end', '--end', default = 200, type = int, help = 'end')

            args = parser.parse_args(sys.argv[2:len(sys.argv)])

            print('data =', args.data)
            print('batch_size =', args.batch_size)
            print('d_model =', args.d_model)
            print('d_ff =', args.d_ff)
            print('n_layers =', args.n_layers)
            print('n_heads =', args.n_heads)
            print('dropout =', args.dropout)
            print('epochs =', args.epochs)

            prefix = args.logs + '-' + str(args.begin) + '-' + str(args.end)

            if not args.gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            if args.data == 'autort':
                (x_train, y_train), (x_val, y_val) = data_autort.load_training_transformer()

                min_val = 0.0
                max_val = 101.33

            elif args.data == 'prosit':
                (x_train, y_train), (x_val, y_val) = data_prosit.load_training()

                min_val = min(y_train.min(), y_val.min()) - 0.01
                max_val = max(y_train.max(), y_val.max()) + 0.01

                y_train = min_max_scale(y_train, min = min_val, max = max_val)
                y_val = min_max_scale(y_val, min = min_val, max = max_val)


            elif args.data == 'deepdia':
                (x_train, y_train), (x_val, y_val) = data_deepdia.load_training_transformer()

                min_val = min(y_train.min(), y_val.min()) - 0.01
                max_val = max(y_train.max(), y_val.max()) + 0.01                                

                y_train = min_max_scale(y_train, min = min_val, max = max_val)
                y_val = min_max_scale(y_val, min = min_val, max = max_val)

            else:
                print('Unknown data')
                exit(0)

            CLS = x_train.max() + 1
            x_train = np.concatenate((np.full((x_train.shape[0], 1), CLS), x_train), axis = 1)
            x_val = np.concatenate((np.full((x_val.shape[0], 1), CLS), x_val), axis = 1)

            print(len(x_train), 'Training sequences')
            print(len(x_val), 'Validation sequences')

            print('CLS =', CLS)

            tf.random.set_seed(1)
            np.random.seed(1)

            # hyperparameters

            HP_N_LAYERS = hp.HParam('n_layers', hp.Discrete(args.n_layers))

            HP_N_HEADS = hp.HParam('n_heads', hp.Discrete(args.n_heads))

            HP_DROPOUT = hp.HParam('dropout', hp.Discrete(args.dropout))

            HP_BATCHSIZE = hp.HParam('batch_size', hp.Discrete(args.batch_size))

            HP_D_MODEL = hp.HParam('d_model', hp.Discrete(args.d_model))

            HP_D_FF = hp.HParam('d_ff', hp.Discrete(args.d_ff))

            METRIC_ACCURACY = 'val_loss'

            with tf.summary.create_file_writer(prefix + '/hparam_tuning').as_default():
                hp.hparams_config(
                    hparams = [HP_N_LAYERS, HP_DROPOUT, HP_N_HEADS, HP_BATCHSIZE, HP_D_MODEL, HP_D_FF],
                    metrics = [hp.Metric(METRIC_ACCURACY, display_name = 'loss')],
                )

            def train_test_model(hparams):

                np.random.seed(1)
                tf.random.set_seed(1)

                model = build_model(
                    num_layers = hparams[HP_N_LAYERS],
                    d_model = hparams[HP_D_MODEL],
                    num_heads = hparams[HP_N_HEADS],
                    d_ff = hparams[HP_D_FF],
                    dropout_rate = hparams[HP_DROPOUT],
                    vocab_size = CLS + 1,           # number of amino acids, including CLS
                    max_len = x_train.shape[1]      # maximal peptide length
                )

                learning_rate = custom_schedule(hparams[HP_D_MODEL])
                optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-9)

                model.compile(optimizer = optimizer, loss = tf.keras.losses.MeanAbsoluteError())

                earlystopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20, verbose=1)

                csv_name = 'model-history-' + args.data + '-b' + str(hparams[HP_BATCHSIZE]) + '-dm' + str(hparams[HP_D_MODEL]) + \
                          '-df' + str(hparams[HP_D_FF]) + '-nl' + str(hparams[HP_N_LAYERS]) + \
                          '-nh' + str(hparams[HP_N_HEADS]) + '-dr' + str(hparams[HP_DROPOUT]) + '-ep' + str(args.epochs) + '.csv'
                csv_logger = tf.keras.callbacks.CSVLogger(csv_name, append = False)

                training_history = model.fit(
                            x_train,
                            y_train,
                            batch_size = hparams[HP_BATCHSIZE],
                            epochs = args.epochs,
                            validation_data = (x_val, y_val),
                            callbacks = [earlystopper, csv_logger],
                            verbose = 1)

                a = pd.read_csv(csv_name)

                return np.min(a['val_loss'])

            def run(run_dir, hparams):
                with tf.summary.create_file_writer(run_dir).as_default():
                    hp.hparams(hparams)  # record the values used in this trial
                    loss = train_test_model(hparams)
                    tf.summary.scalar(METRIC_ACCURACY, loss, step=1)

                return loss

            # create a list of parameters
            import random
            random.seed(args.seed)

            parameter_set = []

            completed = set()

            for session_num in range(args.n_random_samples):

                while True :
                    n_layers = random.sample(HP_N_LAYERS.domain.values, 1)[0]
                    n_heads = random.sample(HP_N_HEADS.domain.values, 1)[0]
                    dropout_rate = random.sample(HP_DROPOUT.domain.values, 1)[0]
                    batch_size = random.sample(HP_BATCHSIZE.domain.values, 1)[0]
                    d_model = random.sample(HP_D_MODEL.domain.values, 1)[0]
                    d_ff = random.sample(HP_D_FF.domain.values, 1)[0]

                    signature = ''.join([str(n_layers), str(n_heads), str(dropout_rate), str(batch_size), str(d_model), str(d_ff)])

                    if signature not in completed:
                        completed.add(signature)
                        break

                hparams = {
                            HP_N_LAYERS: n_layers,
                            HP_N_HEADS: n_heads,
                            HP_DROPOUT: dropout_rate,
                            HP_BATCHSIZE: batch_size,
                            HP_D_MODEL: d_model,
                            HP_D_FF: d_ff,
                }

                print({h.name: hparams[h] for h in hparams})
                parameter_set.append(hparams)

            

            with open(prefix + '.txt', 'w') as myfile:
                myfile.write('N\td_model\td_ff\th\tr\tbatch_size\tloss\n')

            for session_num in range(args.begin, args.end):
                hparams = parameter_set[session_num]
                run_name = 'run-%d' % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                loss = run(prefix + '/hparam_tuning/' + run_name, hparams)

                with open(prefix + '.txt', 'a') as myfile:
                    myfile.write(str(hparams[HP_N_LAYERS]) + '\t' +
                                 str(hparams[HP_D_MODEL]) + '\t' +
                                 str(hparams[HP_D_FF]) + '\t' +
                                 str(hparams[HP_N_HEADS]) + '\t' +
                                 str(hparams[HP_DROPOUT]) + '\t' +
                                 str(hparams[HP_BATCHSIZE]) + '\t' +
                                 str(loss) + '\n')


        elif mode == 'predict':

            if len(sys.argv) < 3:
                print('python rt.py predict model_directory ...')
                sys.exit(0)

            model_path = sys.argv[2]

            parser = argparse.ArgumentParser()
            parser.add_argument('-data', '--data', default = '', type = str, help = 'Data for testing, default empty for using -input.')
            parser.add_argument('-input', '--input', default = '', type = str, help = 'Data for testing, default empty.')
            parser.add_argument('-header', '--header', default = 'sequence', type = str, help = 'Header for column containing peptide sequences')
            parser.add_argument('-epochs', '--epochs', default = 0, type = int, help = 'Number of epochs, default 0.')
            parser.add_argument('--gpu', default=True, action=argparse.BooleanOptionalAction)
            parser.add_argument('-output', '--output', default = 'output.txt', type = str, help = 'prediction output, default output.txt.')

            args = parser.parse_args(sys.argv[3:len(sys.argv)])

            print('data =', args.data)
            print('input =', args.input)
            print('epochs =', args.epochs)
            print('gpu =', args.gpu)
            print('output =', args.output)

            # gpu
            if not args.gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            # epoch
            if args.epochs == 0 :
                a = pd.read_csv(model_path + 'model_history_log.csv')
                model_epoch = np.argmin(a['val_loss']) + 1
            else:
                model_epoch = args.epochs

            # load parameters
            para = pd.read_csv(model_path + 'parameters.txt', sep = '\t', index_col = 0)

            print(para)

            min_val = float(para.loc['min_val', 'value'])
            max_val = float(para.loc['max_val', 'value'])
            model_data = para.loc['data', 'value']

            # test data
            if args.data == 'autort':
                x_test, y_test = data_autort.load_testing_transformer()

            elif args.data == 'prosit':
                x_test, y_test = data_prosit.load_testing()

            elif args.data == 'deepdia':
                x_test, y_test = data_deepdia.load_testing_transformer()

            elif args.data == '':
                if para.loc['data', 'value'] == 'prosit':
                    x_test, y_test = data_generics.load_prosit(args.input, seq_header = args.header)                
                elif para.loc['data', 'value'] == 'deepdia':
                    x_test, y_test = data_generics.load_deepdia(args.input, seq_header = args.header)                
                elif para.loc['data', 'value'] == 'autort':
                    x_test, y_test = data_generics.load_autort(args.input, seq_header = args.header)
                elif para.loc['data', 'value'] == 'phospho':
                    x_test, y_test = data_generics.load_phospho(args.input, seq_header = args.header)
                else:
                    print('Unknown model')
                    exit(0)
            else:
                print('Unknown data')
                exit(0)

            if (args.data == '' and para.loc['data', 'value'] == 'phospho'):
                all_peps = data_generics.integer_to_sequence_phospho(x_test)
            else:    
                all_peps = data_generics.integer_to_sequence(x_test)

            vocab_size = int(para.loc['vocab_size', 'value'])
            x_test = np.concatenate((np.full((x_test.shape[0], 1), vocab_size - 1), x_test), axis = 1) # CLS is vocab_size-1
            print(len(x_test), 'test sequences...')

            model = build_model(
                num_layers = int(para.loc['n_layers', 'value']),
                d_model = int(para.loc['d_model', 'value']),
                num_heads = int(para.loc['n_heads', 'value']),
                d_ff = int(para.loc['d_ff', 'value']),
                dropout_rate = float(para.loc['dropout', 'value']),
                vocab_size = vocab_size,                            # number of aminoacids + 1
                max_len = int(para.loc['max_length', 'value'])      # maximal peptide length
            )

            model.summary()

            np.random.seed(1)
            tf.random.set_seed(1)

            model.load_weights(model_path + '/weights.' + f'{model_epoch:04d}' + '.hdf5')

            y_predict = model.predict(x_test)
            y_predict = y_predict.astype(np.float32).reshape(-1)
            

            # quick test
            if para.loc['data', 'value'] == 'autort':
                a = min_max_scale_rev(y_test, min = 0.0, max = 101.33)
                b = min_max_scale_rev(y_predict, min = 0.0, max = 101.33)                

            elif para.loc['data', 'value'] == 'prosit':
                y_predict = min_max_scale_rev(y_predict, min = min_val, max = max_val)
                
                v = 1883.0160689
                m = 56.35363441
                a = y_test.astype(np.float32).reshape(-1) * np.sqrt(v) + m
                b = y_predict * np.sqrt(v) + m
                
            elif para.loc['data', 'value'] == 'deepdia':
                
                y_predict = min_max_scale_rev(y_predict, min = min_val, max = max_val)

                a = y_test * 100
                b = y_predict * 100

            elif para.loc['data', 'value'] == 'phospho':
                
                y_predict = min_max_scale_rev(y_predict, min = min_val, max = max_val)

                a = y_test
                b = y_predict
                
            print('\nModel epoch =', model_epoch, '; MAE =', np.median(np.abs(a-b)), '\n\n')            

            pd.DataFrame({'sequence': all_peps,
                          'y': y_test.astype(np.float32).reshape(-1),
                          'y_pred': b}).to_csv(args.output, sep = '\t', index = False)

        else:
            print('train or predict or tune')

if __name__ == '__main__':
    main()

