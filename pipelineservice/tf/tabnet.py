import numpy as np
import tensorflow as tf

def sparsemax(logits, axis: int = -1):
    r"""Sparsemax activation function.
    For each batch $i$, and class $j$,
    compute sparsemax activation function:
    $$
    \mathrm{sparsemax}(x)[i, j] = \max(\mathrm{logits}[i, j] - \tau(\mathrm{logits}[i, :]), 0).
    $$
    See [From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification](https://arxiv.org/abs/1602.02068).
    Usage:
    >>> x = tf.constant([[-1.0, 0.0, 1.0], [-5.0, 1.0, 2.0]])
    >>> tfa.activations.sparsemax(x)
    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[0., 0., 1.],
           [0., 0., 1.]], dtype=float32)>
    Args:
        logits: A `Tensor`.
        axis: `int`, axis along which the sparsemax operation is applied.
    Returns:
        A `Tensor`, output of sparsemax transformation. Has the same type and
        shape as `logits`.
    Raises:
        ValueError: In case `dim(logits) == 1`.
    """
    logits = tf.convert_to_tensor(logits, name="logits")

    # We need its original shape for shape inference.
    shape = logits.get_shape()
    rank = shape.rank
    is_last_axis = (axis == -1) or (axis == rank - 1)

    if is_last_axis:
        output = _compute_2d_sparsemax(logits)
        output.set_shape(shape)
        return output

    # If dim is not the last dimension, we have to do a transpose so that we can
    # still perform softmax on its last dimension.

    # Swap logits' dimension of dim and its last dimension.
    rank_op = tf.rank(logits)
    axis_norm = axis % rank
    logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))

    # Do the actual softmax on its last dimension.
    output = _compute_2d_sparsemax(logits)
    output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))

    # Make shape inference work since transpose may erase its static shape.
    output.set_shape(shape)
    return output


def _swap_axis(logits, dim_index, last_index, **kwargs):
    return tf.transpose(
        logits,
        tf.concat(
            [
                tf.range(dim_index),
                [last_index],
                tf.range(dim_index + 1, last_index),
                [dim_index],
            ],
            0,
        ),
        **kwargs,
    )


def _compute_2d_sparsemax(logits):
    """Performs the sparsemax operation when axis=-1."""
    shape_op = tf.shape(logits)
    obs = tf.math.reduce_prod(shape_op[:-1])
    dims = shape_op[-1]

    # In the paper, they call the logits z.
    # The mean(logits) can be substracted from logits to make the algorithm
    # more numerically stable. the instability in this algorithm comes mostly
    # from the z_cumsum. Substacting the mean will cause z_cumsum to be close
    # to zero. However, in practise the numerical instability issues are very
    # minor and substacting the mean causes extra issues with inf and nan
    # input.
    # Reshape to [obs, dims] as it is almost free and means the remanining
    # code doesn't need to worry about the rank.
    z = tf.reshape(logits, [obs, dims])

    # sort z
    z_sorted, _ = tf.nn.top_k(z, k=dims)

    # calculate k(z)
    z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
    z_check = 1 + k * z_sorted > z_cumsum
    # because the z_check vector is always [1,1,...1,0,0,...0] finding the
    # (index + 1) of the last `1` is the same as just summing the number of 1.
    k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)

    # calculate tau(z)
    # If there are inf values or all values are -inf, the k_z will be zero,
    # this is mathematically invalid and will also cause the gather_nd to fail.
    # Prevent this issue for now by setting k_z = 1 if k_z = 0, this is then
    # fixed later (see p_safe) by returning p = nan. This results in the same
    # behavior as softmax.
    k_z_safe = tf.math.maximum(k_z, 1)
    indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
    tau_sum = tf.gather_nd(z_cumsum, indices)
    tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)

    # calculate p
    p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
    # If k_z = 0 or if z = nan, then the input is invalid
    p_safe = tf.where(
        tf.expand_dims(
            tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
            axis=-1,
        ),
        tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
        p,
    )

    # Reshape back to original size
    p_safe = tf.reshape(p_safe, shape_op)
    return p_safe

def glu(act, n_units):
    return act[:, :n_units] * tf.nn.sigmoid(act[:, n_units:])
class TransformBlock(tf.keras.Model):
    def __init__(self,
                 features,
                 norm_type,
                 momentum = 0.9,
                 virtual_batch_size = None,
                 groups = 2,
                 block_name = '',
                 **kwargs
                 ):
        super().__init__()
        self.transform = tf.keras.layers.Dense(features, use_bias = False, name = f'transformblock_dense_{block_name}')
        if norm_type == 'batch':
            self.bn = tf.keras.layers.BatchNormalization(axis = -1, momentum = momentum,
                                                         virtual_batch_size = virtual_batch_size,
                                                         name = f'transformerblock_bn_{block_name}')

    def call(self, x, training = None):
        x = self.transform(x)
        x = self.bn(x, training = training)
        return x




class TabNet(tf.keras.Model):
    def __init__(self,
                 columns,
                 feature_dim = 64,
                 output_dim = 64,
                 num_features=None,
                 num_decision_steps=5,
                 relaxation_factor=1.5,
                 momentum=0.98,
                 virtual_batch_size=None,
                 sparsity_coefficient=1e-5,
                 norm_type='batch',
                 num_groups=2,
                 epsilon=1e-5,

                 ):
        super().__init__()
        if columns is not None:
            try:
                self.num_features = len(columns)
            except:
                self.num_features = int(columns)
        self.num_decision_steps = num_decision_steps
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.virtual_batch_size = virtual_batch_size
        self.relaxation_factor = relaxation_factor
        self.epsilon = epsilon
        if columns is not None:
            if isinstance(columns, list):
                self.input_features = tf.keras.layers.DenseFeatures(columns, trainable=True)
            else:
                self.input_features = tf.keras.layers.Dense(columns)
            if norm_type == 'batch':
                self.input_bn = tf.keras.layers.BatchNormalization(axis = -1, momentum = momentum, name = 'input_bn')
        else:
            self.input_features = None
            self.input_bn = None
        self.transform_f1 = TransformBlock(features = 2*feature_dim,
                                           norm_type = norm_type,
                                           momentum = momentum,
                                           virtual_batch_size = virtual_batch_size,
                                           groups = num_groups,
                                           block_name = 'f1'
                                           )
        self.transform_f2 = TransformBlock(features = 2*feature_dim,
                                           norm_type = norm_type,
                                           momentum = momentum,
                                           virtual_batch_size = virtual_batch_size,
                                           groups = num_groups,
                                           block_name = 'f2'
                                           )
        self.transform_f3_list = [
            TransformBlock(features = 2*feature_dim,
                                           norm_type = norm_type,
                                           momentum = momentum,
                                           virtual_batch_size = virtual_batch_size,
                                           groups = num_groups,
                                           block_name = f'f3_{i}'
                                           )
            for i in range(self.num_decision_steps)
        ]
        self.transform_f4_list = [
            TransformBlock(features = 2*feature_dim,
                                           norm_type = norm_type,
                                           momentum = momentum,
                                           virtual_batch_size = virtual_batch_size,
                                           groups = num_groups,
                                           block_name = f'f4_{i}'
                                           )
            for i in range(self.num_decision_steps)
        ]
        self.transform_coef_list = [
            TransformBlock(features = self.num_features,
                                           norm_type = norm_type,
                                           momentum = momentum,
                                           virtual_batch_size = virtual_batch_size,
                                           groups = num_groups,
                                           block_name = f'coef_{i}'
                                           )
            for i in range(self.num_decision_steps)
        ]

    def call(self, x, is_training = None):
        if self.input_features is not None:
            features = self.input_features(x)
            features = self.input_bn(features, training = is_training)
        else:
            features = inputs
        batch_size = tf.shape(features)[0]

        # Initializes decision-step dependent variables.
        output_aggregated = tf.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = tf.zeros([batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        complemantary_aggregated_mask_values = tf.ones([batch_size, self.num_features])
        total_entropy = 0

        if is_training:
            v_b = self.virtual_batch_size
        else:
            v_b = 1


        for ni in range(self.num_decision_steps):
            transform_f1 = self.transform_f1(features, training = is_training)
            transform_f1 = glu(transform_f1, self.feature_dim)

            transform_f2 = self.transform_f2(transform_f1, training = is_training)
            transform_f2 = (glu(transform_f2, self.feature_dim) + transform_f1)* tf.math.sqrt(0.5)

            transform_f3 = self.transform_f3_list[ni](transform_f2, training = is_training)
            transform_f3 = (glu(transform_f3, self.feature_dim) + transform_f2) * tf.math.sqrt(0.5)

            transform_f4 = self.transform_f4_list[ni](transform_f3, training = is_training)
            transform_f4 = (glu(transform_f4, self.feature_dim) + transform_f3) * tf.math.sqrt(0.5)

            if ni > 0:
                decision_out = tf.nn.relu(transform_f4[:, :self.output_dim])
                output_aggregated +=decision_out
                scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True) / (self.num_decision_steps - 1)
                aggregated_mask_values += mask_values * scale_agg
            features_for_coef = transform_f4[:, self.output_dim:]
            if ni < (self.num_decision_steps - 1):
                mask_values = self.transform_coef_list[ni](features_for_coef, training = is_training)
                mask_values *= complemantary_aggregated_mask_values
                mask_values = sparsemax(mask_values)

                # Relaxation factor controls the amount of reuse of features between
                # different decision blocks and updated with the values of
                # coefficients.
                complemantary_aggregated_mask_values *= (self.relaxation_factor - mask_values)

                # Entropy is used to penalize the amount of sparsity in feature
                # selection.
                total_entropy += tf.reduce_mean(tf.reduce_sum(-mask_values * tf.math.log(mask_values + self.epsilon),axis=1)) / (self.num_decision_steps - 1)
                # Feature selection.
                masked_features = tf.multiply(mask_values, features)
        return output_aggregated, total_entropy

if __name__ == '__main__':
    inputs = np.random.randn(16, 400)
    model  = TabNet(columns = 400)
    out, loss = model(inputs)
    print('output: ', out.shape, loss)
