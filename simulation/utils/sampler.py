import numpy as np
import tensorflow as tf


def sample_selection(g, eg):
    ng = tf.norm(g)
    neg = tf.norm(eg, axis=1)
    mean_sim = tf.matmul(g, eg, transpose_b = True) / tf.maximum(ng * neg, tf.ones_like(neg) * 1e-6)
    negd = tf.expand_dims(neg, 1)
    cross_div = tf.matmul(eg, eg, transpose_b = True) / tf.maximum(tf.matmul(negd, negd, transpose_b = True), tf.ones_like(negd) * 1e-6)
    avg_div = tf.reduce_mean(cross_div, axis=0)
    measure_init = mean_sim - avg_div
    measure = tf.reshape(measure_init, -1)
    _, u_idx = tf.nn.top_k(measure, k=tf.shape(measure)[0], sorted=True)
    return u_idx.numpy()


def select_coreset(X, y, has_factorize, model):
    if not has_factorize:
        with tf.GradientTape() as tape:
            pred = model(X, has_factorize)
            loss = tf.losses.mean_squared_error(pred, y)
        gradients = tape.gradient(loss, model.shared_causal_net.trainable_variables)
        total_grads = []
        for grad in gradients:
            total_grads.append(tf.reshape(grad, shape=(-1,)))

        grads = []
        for i in range(X.shape[0]):
            with tf.GradientTape() as tape2:
                pred_next_state = model(X[i:i+1], has_factorize)
                mse_loss = tf.losses.mean_squared_error(y[i].reshape(1, -1), pred_next_state)
            gradients = tape2.gradient(mse_loss, model.shared_causal_net.trainable_variables)

            res = tf.concat([tf.reshape(g, shape=[-1]).numpy() for g in gradients], axis=0)
            grads.append(res)

    else:
        gradients_ = []
        for i in range(model.state_dim):
            with tf.GradientTape() as tape:
                pred = model(X, has_factorize, i)
                loss = tf.losses.mean_squared_error(pred, y[:, i])
            
            gradients_.append(tape.gradient(loss, model.causal_net[i].trainable_variables))
        total_grads = []
        for i in range(model.state_dim):
            for idx, grad in enumerate(gradients_[i]):
                if i == 0:
                    total_grads.append(tf.reshape(grad, shape=(-1,)))
                else:
                    total_grads[idx] += tf.reshape(grad, shape=(-1,))
        for item in total_grads:
            item /= model.state_dim

        grads = []
        for i in range(X.shape[0]):
            gradients_ = []
            for j in range(model.state_dim):
                with tf.GradientTape() as tape2:
                    pred_next_state = model(X[i:i+1], has_factorize, j)
                    mse_loss = tf.losses.mean_squared_error(y[i, j].reshape(1, -1), pred_next_state)

                gradients_.append(tape2.gradient(mse_loss, model.causal_net[j].trainable_variables))

            tmp_grads = []
            for i in range(model.state_dim):
                for idx, grad in enumerate(gradients_[i]):
                    if i == 0:
                        tmp_grads.append(tf.reshape(grad, shape=(-1,)))
                    else:
                        tmp_grads[idx] += tf.reshape(grad, shape=(-1,))
            for item in tmp_grads:
                item /= model.state_dim

            res = tf.concat([tf.reshape(g, shape=[-1]).numpy() for g in tmp_grads], axis=0)
            grads.append(res)

    # Coreset update
    _g_reshaped = tf.concat(total_grads, axis=0)
    _g = tf.reshape(_g_reshaped, (1, -1))
    _eg = tf.convert_to_tensor(np.array(grads))
    sorted_idx = sample_selection(_g, _eg)
    return sorted_idx
