# Final version of simplicial agent for publication
#
# This code is the v8 agent code, with the relational part removed
# and some cleanup of comments and dead code

import numpy as np
import math

import tensorflow as tf
from keras import layers
from keras.layers import Dense, Flatten, Concatenate
from keras import backend as K

from ray.rllib.models import Model
from keras_transformer.attention import _BaseMultiHeadAttention
from keras_transformer.transformer import LayerNormalization, TransformerTransition

# Based on MultiHeadSelfAttention from Keras-RL
# https://github.com/kpot/keras-transformer/blob/master/keras_transformer/attention.py
class MultiHeadSelfAttentionZambaldi(_BaseMultiHeadAttention):
    """
    Multi-head self-attention for both encoders and decoders.
    Uses only one input and has implementation which is better suited for
    such use case that more general MultiHeadAttention class.
    """
    def __init__(self, num_virtual_entities, **kwargs):
        self.num_virtual_entities = num_virtual_entities
        
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
    
    def attention_zambaldi(self, pre_q, pre_v, pre_k, seq_len: int, d_model: int,
                  training=None):
        """
        Calculates the output of the attention once the affine transformations
        of the inputs are done. Here's the shapes of the arguments:
        :param pre_q: (batch_size, q_seq_len, num_heads, d_model // num_heads)
        :param pre_v: (batch_size, v_seq_len, num_heads, d_model // num_heads)
        :param pre_k: (batch_size, k_seq_len, num_heads, d_model // num_heads)
        :param out_seq_len: the length of the output sequence
        :param d_model: dimensionality of the model (by the paper)
        """
        
        d_submodel = d_model // self.num_heads
        num_ve = self.num_virtual_entities
        
        # shaping Q and V into (batch_size, num_heads, seq_len, d_model//heads)
        q = K.permute_dimensions(pre_q, [0, 2, 1, 3])
        k = K.permute_dimensions(pre_k, [0, 2, 1, 3])
        v = K.permute_dimensions(pre_v, [0, 2, 1, 3])

        # collapse the batch dimension and head dimensions so we can operate
        # simultaneously on all heads, the following have shape (-1, seq_len, d_submodel)
        q = K.reshape(q, (-1,seq_len,d_submodel))
        k = K.reshape(k, (-1,seq_len,d_submodel))
        v = K.reshape(v, (-1,seq_len,d_submodel))
        
        # Only standard entities update representations of standard entities, but
        # virtual entities receive updates from all entities
        qk_standard = tf.einsum('aib,ajb->aij', q[:, :seq_len-num_ve, :], k[:, :seq_len-num_ve, :]) # (-1, seq_len-num_ve, seq_len-num_ve)
        qk_virtual  = tf.einsum('aib,ajb->aij', q[:, seq_len-num_ve:, :], k) # (-1, num_ve, seq_len)

        sqrt_d = K.constant(np.sqrt(d_submodel), dtype=K.floatx())
        a_standard = K.softmax( qk_standard / sqrt_d ) # (-1, seq_len-num_ve, seq_len-num_ve)
        a_virtual  = K.softmax( qk_virtual / sqrt_d ) # (-1, num_ve, seq_len)
        
        av_standard = tf.einsum('aij,ajc->aic', a_standard, v[:,:seq_len-num_ve,:]) # (-1, seq_len-num_ve, d_submodel)
        av_virtual = tf.einsum('aij,ajc->aic', a_virtual, v) # (-1, num_ve, d_submodel)
        av = K.concatenate( [av_standard, av_virtual], axis=-2 ) # (-1, seq_len, d_submodel)
        
        attention_heads = K.reshape( av, (-1, self.num_heads, seq_len, d_submodel))
        attention_heads = K.permute_dimensions(attention_heads, [0, 2, 1, 3]) # (-1, seq_len, self.num_heads, d_submodel)
        attention_heads_concatenated = K.reshape(attention_heads,(-1,seq_len,d_model))
        
        return attention_heads_concatenated
        
def multi_softmax(target, axis, name=None):
  with tf.name_scope(name, 'softmax', values=[target]):
    max_axis = tf.reduce_max(target, axis, keep_dims=True)
    target_exp = tf.exp(target-max_axis)
    normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
    softmax = target_exp / normalize
    return softmax
    
class MultiHeadSelfAttentionSimplicial(_BaseMultiHeadAttention):
    """
    Multi-head self-attention for both encoders and decoders.
    Uses only one input and has implementation which is better suited for
    such use case that more general MultiHeadAttention class.
    """
    def __init__(self, num_heads: int, d_simp_model: int,
                 num_virtual_entities: int,
                 use_masking: bool,
                 dropout: float = 0.0,
                 compression_window_size: int = None,
                 **kwargs):
        self.d_simp_model = d_simp_model
        self.num_virtual_entities = num_virtual_entities
        
        super().__init__(num_heads=num_heads, 
                        use_masking=use_masking, 
                        dropout=dropout,
                        compression_window_size=compression_window_size,
                        **kwargs)
    
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

        self.qkkv_weights = self.add_weight(
            name='qkkv_weights',
            shape=(d_model, self.d_simp_model * 4),  # * 4 for q, k1, k2 and v
            initializer='glorot_uniform',
            trainable=True)
        
        # TODO: at the moment this code is assuming we have only one-head of 
        # 2-simplicial attention
        self.B_weights = self.add_weight(
            name='B_weight',
            shape=(self.d_simp_model // self.num_heads,
                    self.d_simp_model // self.num_heads,
                    self.d_simp_model // self.num_heads),
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
        qkkv = K.dot(inputs, self.qkkv_weights) # (-1,seq_len,d_simp_model*4)
        qkkv = K.reshape(qkkv, [-1,self.d_simp_model*4])

        # splitting the keys, the values and the queries before further
        # processing
        pre_q, pre_k1, pre_k2, pre_v = [
            K.reshape(
                # K.slice(qkv, (0, i * d_model), (-1, d_model)),
                qkkv[:, i * self.d_simp_model:(i + 1) * self.d_simp_model],
                (-1, seq_len, self.num_heads, self.d_simp_model // self.num_heads))
            for i in range(4)]
        
        attention_out = self.attention_simplicial(pre_q, pre_v, pre_k1, pre_k2, seq_len,
                                        self.d_simp_model,
                                       training=kwargs.get('training'))
                                       
        # of shape (-1, seq_len, d_model)
        return attention_out

    def compute_output_shape(self, input_shape):
        shape_a, seq_len, d_model = input_shape
        return (shape_a, seq_len, self.d_simp_model)
    
    def attention_simplicial(self, pre_q, pre_v, pre_k1, pre_k2, seq_len: int, d_simp_model: int,
                  training=None):
        d_submodel = d_simp_model // self.num_heads
        num_ve = self.num_virtual_entities

        # shaping Q,V,K1,K2 into (batch_size, num_heads, seq_len, d_submodel)
        q = K.permute_dimensions(pre_q, [0, 2, 1, 3])
        v = K.permute_dimensions(pre_v, [0, 2, 1, 3])
        k1 = K.permute_dimensions(pre_k1, [0, 2, 1, 3])
        k2 = K.permute_dimensions(pre_k2, [0, 2, 1, 3])
        
        v_virtual_original = v[:,:,seq_len-num_ve:,:] # (batch_size, num_heads, seq_len, d_submodel)

        # collapse the batch dimension and head dimensions so we can operate
        # simultaneously on all heads, the following have shape (-1, seq_len, d_submodel)
        q = K.reshape(q, (-1,seq_len,d_submodel))
        v = K.reshape(v, (-1,seq_len,d_submodel))
        k1 = K.reshape(k1, (-1,seq_len,d_submodel))
        k2 = K.reshape(k2, (-1,seq_len,d_submodel))
        
        # We generate queries only for standard entities, and keys and
        # values only for virtual entities
        q = q[:, :seq_len-num_ve, :] # shape (-1,seq_len-NUM_VIRTUAL_ENTITIES,d_simp_model//heads)
        v = v[:, seq_len-num_ve:, :] # shape (-1,NUM_VIRTUAL_ENTITIES,d_simp_model//heads)
        k1 = k1[:, seq_len-num_ve:, :] # shape (-1,NUM_VIRTUAL_ENTITIES,d_simp_model//heads)
        k2 = k2[:, seq_len-num_ve:, :] # shape (-1,NUM_VIRTUAL_ENTITIES,d_simp_model//heads)
        
        # (q k1 k2)_1 = (q . k1) k2 - (q . k2) k1 + (k1 . k2) q
        
        # NOTE: q comes from object i, k1 comes from object j, and k2 comes from object k
        # so we mark the seq_len dimensions with these indices so we don't mix them up

        # note the dummy dimension z in self.proj_weights is summed over (i.e. removed)
        qk1 = tf.einsum('aib,ajb->aij', q, k1)
        qk2 = tf.einsum('aib,akb->aik', q, k2)
        k1k2 = tf.einsum('ajb,akb->ajk', k1, k2)

        # || (q k1 k2)_1 ||^2 = (q.k1)^2(k2.k2) + (k1.k2)^2(q.q) + (q.k2)^2(k1.k1) 
        #                       - 2(q.k1)(q.k2)(k1.k2)  
        #
        # pre_logitsvector = || (q k1 k2)_1 ||^2

        k2k2 = tf.einsum('akc,akc->ak', k2, k2)
        qq = tf.einsum('aib,aib->ai', q, q)
        k1k1 = tf.einsum('ajb,ajb->aj', k1, k1)
        
        qk1k2k2 = tf.einsum('aij,ak->aijk', K.square(qk1), k2k2)
        k1k2qq = tf.einsum('ajk,ai->aijk', K.square(k1k2), qq)
        qk2k1k1 = tf.einsum('aik,aj->aijk', K.square(qk2), k1k1)
        
        qk1_e = K.expand_dims(qk1, axis=3) # qk1_e = tf.einsum('aij->aijk',qk1)
        qk2_e = K.expand_dims(qk2, axis=2) # qk2_e = tf.einsum('aik->aijk',qk2)
        k1k2_e = K.expand_dims(k1k2, axis=1) # k1k2_e = tf.einsum('ajk->aijk',k1k2)

        pre_logitsvector = qk1k2k2 + k1k2qq + qk2k1k1 - 2 * qk1_e * qk2_e * k1k2_e
        logitsvector = K.sqrt( pre_logitsvector )
                
        # In an equation, with \mu, \sigma the mean and standard deviation over all
        # entries of the tensor logitsvector, we have
        #
        # logitsvector_norm_{i,j,k} = gain_{i}/\sigma( logitsvector_{i,j,k} - \mu) + bias_{i}
        #
        
        a = multi_softmax( logitsvector, axis=[-2,-1]) # shape (-1, seq_len-NUM_VIRTUAL_ENTITIES,
                                                       # NUM_VIRTUAL_ENTITIES, NUM_VIRTUAL_ENTITIES)
                                                       # this computes p^{i}_{jk}
      
        # computes \sum_{j,k}p^{i}_{j,k} B( v[j] \otimes v[k] )
        # where p^{i}_{j,k} is the probability of a 2-simplex with vertices (i,j,k)
        # attention_heads has shape (-1,seq_len,d_simp_model//heads)
        Bvj = tf.einsum('qrs,ajr->aqsj',self.B_weights,v)
        Bvjvk = tf.einsum('aqsj,aks->aqjk',Bvj,v)

        attention_heads = tf.einsum('aijk,aqjk->aiq', a, Bvjvk)

        attention_heads = K.reshape(attention_heads,(-1, self.num_heads, seq_len-num_ve, d_submodel))
        attention_heads = K.concatenate([attention_heads,v_virtual_original],axis=-2)
        attention_heads = K.permute_dimensions(attention_heads, [0, 2, 1, 3]) # (-1, seq_len, self.num_heads, d_simp_model//heads)
        attention_heads_concatenated = K.reshape(attention_heads,(-1,seq_len,d_simp_model))
        
        # of shape (-1, seq_len, d_simp_model)
        return attention_heads_concatenated
 
class SimplicialModel(Model):

    def _build_layers_v2(self, input_dict, num_outputs, options):
        
        TRANSFORMER_SIMPLICIAL_DIM = options["custom_options"]["transformer_simplicial_model_dim"]
        TRANSFORMER_MODEL_DIM = options["custom_options"]["transformer_model_dim"]
        TRANSFORMER_STYLE = options["custom_options"]["transformer_style"]
        TRANSFORMER_NUM_HEADS = options["custom_options"]["transformer_num_heads"]
        TRANSFORMER_DEPTH = options["custom_options"]["transformer_depth"]
        CONV_PADDING = options["custom_options"]["conv_padding"]
        NUM_VIRTUAL_ENTITIES = options["custom_options"]["num_virtual_entities"]
       
        # For detailed comments see the base agent
        
        inputs = input_dict["obs"]
        
        sess = tf.get_default_session()
        K.set_session(sess)

        attention_layer = MultiHeadSelfAttentionZambaldi(name='self_attention',
            num_heads=TRANSFORMER_NUM_HEADS, use_masking = False, dropout = 0,
            compression_window_size = None, num_virtual_entities=NUM_VIRTUAL_ENTITIES)
        attention_layer_2simplex = MultiHeadSelfAttentionSimplicial(name='self_2attention',
            num_heads=1, d_simp_model=TRANSFORMER_SIMPLICIAL_DIM, use_masking = False, dropout = 0,
            compression_window_size = None, num_virtual_entities=NUM_VIRTUAL_ENTITIES)
        dense_layer1 = layers.Dense(TRANSFORMER_MODEL_DIM,activation='relu') 
        dense_layer2 = layers.Dense(TRANSFORMER_MODEL_DIM)
                
        def transformer_block(input):
            a = LayerNormalization()(input)

            a1 = attention_layer(a) # a1 = attention(h) has shape -1, seq_len, TRANSFORMER_MODEL_DIM
            a2 = attention_layer_2simplex(a) # shape -1, seq_len, TRANSFORMER_SIMPLICIAL_DIM
            
            a2 = LayerNormalization()(a2)
            
            ac = Concatenate()([a1,a2]) # shape -1, seq_len, TRANSFORMER_MODEL_DIM + TRANSFORMER_SIMPLICIAL_DIM
            b = dense_layer1(ac) 
            b2 = dense_layer2(b) # b = ff(ac) 
            r = layers.Add()([input,b2])
            Hprime = LayerNormalization()(r)
              
            return Hprime
        
        # CONVOLUTIONS ------
        #
        x = layers.Lambda(lambda x: x / 255)(inputs) # rescale RGB to [0,1]
        x = layers.Conv2D(12,(2,2),activation='relu',padding=CONV_PADDING)(x)
        x = layers.Conv2D(24,(2,2),activation='relu',padding=CONV_PADDING)(x) # output shape -1, num_rows, num_cols, 62
        x = layers.Dense(TRANSFORMER_MODEL_DIM-2,
                         activation=None,
                         use_bias=False)(x) # output shape -1, num_rows, num_cols, TRANSFORMER_MODEL_DIM-2
        
        # POSITION EMBEDDING -----
        #
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
        x = layers.Reshape((num_rows*num_cols,d_model+2))(x) # shape (None, num_rows*num_cols,d_model+2)
        
        # NOTE: the batch dimension is preserved by reshape, see https://www.tensorflow.org/api_docs/python/tf/keras/layers/Reshape
        
        # We now add some virtual entities, which are initialised randomly
        tokens = np.arange(NUM_VIRTUAL_ENTITIES).reshape((1,NUM_VIRTUAL_ENTITIES)) # [[0,1,2,...,NUM_VIRTUAL_ENTITIES]]
        tokens = K.constant(tokens)
        ve = layers.Embedding(input_dim=NUM_VIRTUAL_ENTITIES,
                                output_dim=d_model+2)(tokens) # shape (1,NUM_VIRTUAL_ENTITIES,d_model+2)
        ve_tiled = K.tile(ve,[K.shape(x)[0],1,1])
        x = Concatenate(axis=1)([x,ve_tiled])
        
        # TRANSFORMER -----
        for i in range(TRANSFORMER_DEPTH):
            x = transformer_block(x)
        
        # The output of the simplicial Transformer includes the virtual entities,
        # which we now want to remove. The current tensor is of shape
        # (None,num_rows*num_cols+NUM_VIRTUAL_ENTITIES,TRANSFORMER_MODEL_DIM)
        x = x[:,:-NUM_VIRTUAL_ENTITIES,:]
        
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

        sess = tf.get_default_session()
        K.set_session(sess)
        
        x = layers.Dense(1)(self.last_layer)
        x = tf.keras.backend.squeeze(x,1)

        return x
