# Final version of relational agent for publication
#
# This code is the v2 agent code, with the simplicial part removed
# and some cleanup of comments and dead code

import numpy as np
import math

import tensorflow as tf
from keras import layers
from keras.layers import Dense, Flatten, Concatenate
from keras import backend as K

from ray.rllib.models import Model
from keras_transformer.attention import _BaseMultiHeadAttention
from keras_transformer.transformer import LayerNormalization

# Based on MultiHeadSelfAttention from Keras-RL
# https://github.com/kpot/keras-transformer/blob/master/keras_transformer/attention.py
class MultiHeadSelfAttentionZambaldi(_BaseMultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # noinspection PyAttributeOutsideInit
    def build_output_params(self, d_model):
        # We do not use output_weights so override
        return
            
    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if not isinstance(input_shape, tuple):
            raise ValueError('Invalid input')
        d_model = input_shape[-1]
        
        self.validate_model_dimensionality(d_model)
        # These weights are concatenated matrices W_q, W_k and W_v which
        # are, in turn, concatenated W matrices of keys, queries and values
        # for each of the heads. So, essentially it's a concatenation of
        # W_q1, W_q2,..., W_qh, W_k1, W_k2,..., W_kh, W_v1, W_v2,..., W_vh
        # for all h heads.
        self.qkv_weights = self.add_weight(
            name='qkv_weights',
            shape=(d_model, d_model * 3),  # * 3 for q, k and v
            initializer='glorot_uniform',
            trainable=True)
        
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        if not K.is_tensor(inputs):
            raise ValueError(
                'The layer can be called only with one tensor as an argument')
        _, seq_len, d_model = K.int_shape(inputs)
        
        # The first thing we need to do is to perform affine transformations
        # of the inputs to get the Queries, the Keys and the Values.
        qkv = K.dot(inputs, self.qkv_weights) # (-1,seq_len,d_model*3)
        qkv = K.reshape(qkv,[-1,d_model*3])

        # splitting the keys, the values and the queries before further
        # processing
        pre_q, pre_k, pre_v = [
            K.reshape(
                # K.slice(qkv, (0, i * d_model), (-1, d_model)),
                qkv[:, i * d_model:(i + 1) * d_model],
                (-1, seq_len, self.num_heads, d_model // self.num_heads))
            for i in range(3)]
        
        attention_out = self.attention_zambaldi(pre_q, pre_v, pre_k, seq_len, d_model,
                                       training=kwargs.get('training'))
        # of shape (-1, seq_len, d_model)
        return attention_out

    def compute_output_shape(self, input_shape):
        shape_a, seq_len, d_model = input_shape
        return (shape_a, seq_len, d_model)
    
    def attention_zambaldi(self, pre_q, pre_v, pre_k, out_seq_len: int, d_model: int,
                  training=None):
        """
        Calculates the output of the attention once the affine transformations
        of the inputs are done. Here's the shapes of the arguments:
        :param pre_q: (batch_size, q_seq_len, num_heads, d_model // num_heads)
        :param pre_v: (batch_size, v_seq_len, num_heads, d_model // num_heads)
        :param pre_k: (batch_size, k_seq_len, num_heads, d_model // num_heads)
        :param out_seq_len: the length of the output sequence
        :param d_model: dimensionality of the model (by the paper)
        :param training: Passed by Keras. Should not be defined manually.
          Optional scalar tensor indicating if we're in training
          or inference phase.
        """
        # shaping Q and V into (batch_size, num_heads, seq_len, d_model//heads)
        q = K.permute_dimensions(pre_q, [0, 2, 1, 3])
        v = K.permute_dimensions(pre_v, [0, 2, 1, 3])

        k_transposed = K.permute_dimensions(pre_k, [0, 2, 3, 1])
        
        # shaping K into (batch_size, num_heads, d_model//heads, seq_len)
        # for further matrix multiplication
        sqrt_d = K.constant(np.sqrt(d_model // self.num_heads),
                            dtype=K.floatx())
        q_shape = K.int_shape(q)
        k_t_shape = K.int_shape(k_transposed)
        v_shape = K.int_shape(v)
        # before performing batch_dot all tensors are being converted to 3D
        # shape (batch_size * num_heads, rows, cols) to make sure batch_dot
        # performs identically on all backends
        attention_heads = K.reshape(
            K.batch_dot(
                self.apply_dropout_if_needed(
                    K.softmax(
                        self.mask_attention_if_needed(
                            K.batch_dot(
                                K.reshape(q, (-1,) + q_shape[-2:]),
                                K.reshape(k_transposed,
                                          (-1,) + k_t_shape[-2:]))
                            / sqrt_d)),
                    training=training),
                K.reshape(v, (-1,) + v_shape[-2:])),
            (-1, self.num_heads, q_shape[-2], v_shape[-1]))
        
        # attention_heads has shape (-1, self.num_heads,
        # q_shape[-2] = seq_len, v_shape[-1] = d_model//heads)
        attention_heads = K.permute_dimensions(attention_heads, [0, 2, 1, 3])
        
        # attention_heads now has shape (-1, seq_len, self.num_heads,d_model//heads)
        attention_heads_concatenated = K.reshape(attention_heads,(-1,q_shape[-2],d_model))
        
        # returns the \oplus_h a_i^h from Zambaldi et al
        # of shape (-1, seq_len, d_model)
        return attention_heads_concatenated
    
class BaseModel(Model):
    """Defines an abstract network model for use with RLlib.
    Models convert input tensors to a number of output features. These features
    can then be interpreted by ActionDistribution classes to determine
    e.g. agent action values.
    The last layer of the network can also be retrieved if the algorithm
    needs to further post-processing (e.g. Actor and Critic networks in A3C).
    Attributes:
        input_dict (dict): Dictionary of input tensors, including "obs",
            "prev_action", "prev_reward", "is_training".
        outputs (Tensor): The output vector of this model, of shape
            [BATCH_SIZE, num_outputs].
        last_layer (Tensor): The feature layer right before the model output,
            of shape [BATCH_SIZE, f].
        state_init (list): List of initial recurrent state tensors (if any).
        state_in (list): List of input recurrent state tensors (if any).
        state_out (list): List of output recurrent state tensors (if any).
        seq_lens (Tensor): The tensor input for RNN sequence lengths. This
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        """Define the layers of a custom model.
        Arguments:
            input_dict (dict): Dictionary of input tensors, including "obs",
                "prev_action", "prev_reward", "is_training".
            num_outputs (int): Output tensor must be of size
                [BATCH_SIZE, num_outputs].
            options (dict): Model options.
        Returns:
            (outputs, feature_layer): Tensors of size [BATCH_SIZE, num_outputs]
                and [BATCH_SIZE, desired_feature_size].
        """
        
        TRANSFORMER_MODEL_DIM = options["custom_options"]["transformer_model_dim"]
        TRANSFORMER_NUM_HEADS = options["custom_options"]["transformer_num_heads"]
        TRANSFORMER_DEPTH = options["custom_options"]["transformer_depth"]
        CONV_PADDING = options["custom_options"]["conv_padding"]
        
        # Agent architecture p.15 of Zambaldi et al
        # "The input module contained two convolutional layers with 12 and 24 kernels, 2 × 2 kernel sizes
        # and a stride of 1, followed by a rectified linear unit (ReLU) activation function. The output
        # was tagged with two extra channels indicating the spatial position (x and y) of each cell in
        # the feature map using evenly spaced values between −1 and 1. This was passed to the relational
        # module, consisting of relational blocks, with shared parameters. Queries, keys and values were
        # produced by 2 to 4 attention heads and had an embedding size (d) of 64. The output of this module
        # was aggregated using a feature-wise max pooling function and passed to 4 fully connected layers,
        # each followed by a ReLU. Policy logits (pi, size 4) and baseline function (B, size 1) were produced
        # by a linear projection. The policy logits were normalized and used as multinomial distribution from
        # which the action (a) was sampled."
        
        # NOTE: there is no dropout in Zambaldi et al
        
        inputs = input_dict["obs"]
        
        sess = tf.get_default_session()
        K.set_session(sess)

        # NOTE: the weights in the self-attention mechanism
        # and feed-forward layers are shared between all Transformer blocks (as in
        # Zambaldi et al, but unlike every other Transformer paper)
        
        # The published version of Zambaldi et al does not tell us what the MLP g_\theta is, but
        #
        # - in Santoro et al "A simple neural network module for relational reasoning"
        #   the analogous g_\theta is is a four-layer MLP with 256 dimensional hidden layers
        #   with ReLU non-linearities
        # - in Keras-Transformer the default is a two layer model with hidden dimension
        #   equal to 4 * the embedding dimension, which in the case of 64 dimensional embeddings
        #   gives 256 (this is also the convention in the Sparse Transformer paper)
        # - in the first version of Zambaldi et al they write "passed to a multilayer perceptron
        #   (2-layer MLP with ReLU non-linearities) with the same layers sizes as ei"
        #
        # Hence, attempting to follow Zambaldi, we use layer size TRANSFORMER_MODEL_DIM
        # (in v6 we used 4 times this)
        
        attention_layer = MultiHeadSelfAttentionZambaldi(name='self_attention',
            num_heads=TRANSFORMER_NUM_HEADS, use_masking = False, dropout = 0,
            compression_window_size = None)
        dense_layer1 = layers.Dense(TRANSFORMER_MODEL_DIM,activation='relu')
        dense_layer2 = layers.Dense(TRANSFORMER_MODEL_DIM)
               
        def transformer_block(input):
            #_, seq_len, d_model = K.int_shape(input)
            a = LayerNormalization()(input)
            a = attention_layer(a) # a = attention(h) has shape -1, seq_len, TRANSFORMER_MODEL_DIM
            b = dense_layer1(a) 
            b = dense_layer2(b) # b = ff(a) 
            r = layers.Add()([input,b])
            Hprime = LayerNormalization()(r)
              
            return Hprime
        
        # CONVOLUTIONS ------
        #
        # Question: should we use max-pooling here? It seems not, as the downsampling in the
        # Santoro et al paper "A simple neural network module for relational reasoning"
        # occurs using 3x3 patches with stride 2, rather than max-pooling, and it is not
        # mentioned anywhere in the papers.
        #
        # It is worth comparing to e.g. the models for deep RL on 3D environments in the IMPALA
        # paper, see Figure 3, which also have no max-pooling layers and downsample instead using
        # strided convolutional layers. You'll see there also they prefer hidden layers of width
        # 256 for the FC layers after the initial convolutional layers, in processing visual scenes.
        # So the Zambaldi paper is consistent with their other work on deep RL, in terms of the model.
       
        x = layers.Lambda(lambda x: x / 255)(inputs) # rescale RGB to [0,1]
        x = layers.Conv2D(12,(2,2),activation='relu',padding=CONV_PADDING)(x)
        x = layers.Conv2D(24,(2,2),activation='relu',padding=CONV_PADDING)(x) # output shape -1, num_rows, num_cols, 62
        x = layers.Dense(TRANSFORMER_MODEL_DIM-2,
                         activation=None,
                         use_bias=False)(x) # output shape -1, num_rows, num_cols, TRANSFORMER_MODEL_DIM-2
        
        # NOTE: we are using the default "valid" padding, so actually our width and height decrease
        # by one in each convolutional layer
        
        # POSITION EMBEDDING -----
        #
        # Here we follow Zambaldi et al, rather than the standard Transformer
        # positional embeddings
        
        num_rows, num_cols, d_model = x.get_shape().as_list()[-3:]
        
        ps = np.zeros([num_rows,num_cols,2],dtype=K.floatx()) # shape (12,13,2)
        for ty in range(num_rows):
            for tx in range(num_cols):
                ps[ty,tx,:] = [(2/(num_rows-1))*ty - 1,(2/(num_cols-1))*tx - 1]
        
        ps_expand = K.expand_dims(K.constant(ps),axis=0) # shape (1,num_rows,num_cols,2)
        ps_tiled = K.tile(ps_expand,[K.shape(x)[0],1,1,1]) # shape (None,num_rows,num_cols,2)
        
        # (None,num_rows,num_cols,62) concatenate with (None,num_rows,num_cols,2)
        # to get (None,num_rows,num_cols,TRANSFORMER_MODEL_DIM)
        x = Concatenate(axis=3)([x,ps_tiled])
        x = layers.Reshape((num_rows*num_cols,d_model+2))(x)
        
        # TRANSFORMER -----
        for i in range(TRANSFORMER_DEPTH):
            x = transformer_block(x)

        # MAX-POOLING -----
        # from p.4 "The E~ matrix, with shape Nxf is reudced to an f-dimensional vector by max-pooling
        # over the entity dimension. This pooled vector is then passed to a small MLP..."
        num_entities, d_model = x.get_shape().as_list()[-2:]
        x = layers.MaxPooling1D(pool_size=num_entities)(x)
        x = layers.Flatten()(x)

        # FULLY-CONNECTED LAYERS ----
        x = layers.Dense(256,activation='relu')(x)
        x = layers.Dense(256,activation='relu')(x)
        x = layers.Dense(256,activation='relu')(x)
        x = layers.Dense(256,activation='relu')(x)
        output_tensor = layers.Dense(4)(x) # final output is logits
        
        return output_tensor, x

    def value_function(self):
        """Builds the value function output.
        This method can be overridden to customize the implementation of the
        value function (e.g., not sharing hidden layers).
        Returns:
            Tensor of size [BATCH_SIZE] for the value function.
        """
        
        # NOTE: RLlib can do this automatically, but there seems to be a bug
        # in their implementation, so we rolled one ourself

        sess = tf.get_default_session()
        K.set_session(sess)
        
        x = layers.Dense(1)(self.last_layer)
        x = tf.keras.backend.squeeze(x,1)

        return x