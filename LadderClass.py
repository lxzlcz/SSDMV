# -*- coding: utf-8 -*-
import tensorflow as tf
import input_data
import math
import os
import csv
from tqdm import tqdm

class Ladder(object):
    def __init__(self, label_nums):
        #The count of nodes in each layer in each view#
        self.layer_sizes_p = [512, 1000, 500, 250, 10]
        self.layer_sizes_t = [512, 1000, 500, 250, 10]
        self.layer_sizes_u = [512, 1000, 500, 250, 10]
        self.layer_sizes_d = [3]

        #The layer count of each layer#
        self.L_p = len(self.layer_sizes_p) - 1  # number of layers
        self.L_t = len(self.layer_sizes_t) - 1
        self.L_u = len(self.layer_sizes_u) - 1

        #The number of labeled training samples#
        self.label_nums = label_nums

        #start learning rate#
        self.starter_learning_rate = 0.02

        # epoch after which to begin learning rate decay#
        self.decay_after = 15
        #batch size#
        self.batch_size = 200
        self.noise_std = 0.5  # scaling factor for noise used in corrupted encoder
        
        # hyperparameters that denote the importance of each layer
        self.denoising_cost_p = [1000.0, 10.0, 0.10, 0.10,  0.10]
        self.denoising_cost_t = [1000.0, 10.0, 0.10, 0.10,  0.10]
        self.denoising_cost_u = [1000.0, 10.0, 0.10, 0.10,  0.10]

        #Get labeled samples from a batch#
        self.labeled = lambda x: tf.slice(x, [0, 0], [self.batch_size if self.label_nums >= self.batch_size else self.label_nums, -1]) if x is not None else x
        #Get unlabeled samples from a batch#
        self.unlabeled = lambda x: tf.slice(x, [self.batch_size if self.label_nums >= self.batch_size else self.label_nums, 0], [-1, -1]) if x is not None else x
        #Get labeled & unlabeled samples from a batch#
        self.split_lu = lambda x: (self.labeled(x), self.unlabeled(x))
        
        self.join = lambda l, u: tf.concat([l, u], 0)
        self.model()
        
    #generate the weights and bias for each layer#
    def bi(self, inits, size, name):
        return tf.Variable(inits * tf.ones([size]), name=name)

    def wi(self, shape, name):
        return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])

    def batch_normalization(self, batch, mean=None, var=None):
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=[0])
        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

    def update_batch_normalization(batch, l, running_mean, running_var, bn_assigns, ewma):
        "batch normalize + update average mean and variance of layer l"
        mean, var = tf.nn.moments(batch, axes=[0])
        assign_mean = running_mean[l-1].assign(mean)
        assign_var = running_var[l-1].assign(var)
        bn_assigns.append(ewma.apply([running_mean[l-1], running_var[l-1]]))
        with tf.control_dependencies([assign_mean, assign_var]):
            return (batch - mean) / tf.sqrt(var + 1e-10)
    def training_batch_norm(self, z_pre_l, z_pre_u, l):
        # Training batch normalization
        # batch normalization for labeled and unlabeled examples is performed separately
        if self.noise_std > 0:
            # Corrupted encoder
            # batch normalization + noise
            z = self.join(self.batch_normalization(z_pre_l), self.batch_normalization(z_pre_u))
            z += tf.random_normal(tf.shape(z)) * self.noise_std
        else:
            # Clean encoder
            # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
            z = self.join(self.update_batch_normalization(z_pre_l, l), self.batch_normalization(z_pre_u))
        return z

    def eval_batch_norm(self, ewma, running_mean, running_var, z_pre, l):
        # Evaluation batch normalization
        # obtain average mean and variance and use it to normalize the batch
        mean = ewma.average(running_mean[l-1])
        var = ewma.average(running_var[l-1])
        z = self.batch_normalization(z_pre, mean, var)
        return z

    #encoder, parameter noise_std to control this is a clean encoder or a corrupted encoder
    def encoder(self, inputs, noise_std, weights, training, ewma, running_mean, running_var, L, layer_sizes):
        h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std  # add noise to input

        d = {}  # to store the pre-activation, activation, mean and variance for each layer
        # The data for labeled and unlabeled examples are stored separately
        d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
        d['labeled']['z'][0], d['unlabeled']['z'][0] = self.split_lu(h)
        for l in range(1, L+1):
            print "Layer ", l, ": ", layer_sizes[l-1], " -> ", layer_sizes[l]
            d['labeled']['h'][l-1], d['unlabeled']['h'][l-1] = self.split_lu(h)
            z_pre = tf.matmul(h, weights['W'][l-1])  # pre-activation
            z_pre_l, z_pre_u = self.split_lu(z_pre)  # split labeled and unlabeled examples
            m, v = tf.nn.moments(z_pre_u, axes=[0])

            # perform batch normalization according to value of boolean "training" placeholder:
            z = tf.cond(self.training, lambda: self.training_batch_norm(z_pre_l, z_pre_u, l), lambda: self.eval_batch_norm(ewma, running_mean, running_var, z_pre, l))
            if l == L:
                # use softmax activation in output layer
                h = tf.nn.softmax(weights['gamma'][l-1] * (z + weights["beta"][l-1]))
            else:
                # use ReLU activation in hidden layers
                h = tf.nn.relu(z + weights["beta"][l-1])

            d['labeled']['z'][l], d['unlabeled']['z'][l] = self.split_lu(z)

            d['unlabeled']['m'][l], d['unlabeled']['v'][l] = m, v  # save mean and variance of unlabeled examples for decoding
        d['labeled']['h'][l], d['unlabeled']['h'][l] = self.split_lu(h)
        return h, d

    def g_gauss(self, z_c, u, size):
        "gaussian denoising function proposed in the original paper"
        wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')

        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu
        return z_est



    def decoder(self, clean, corr, y_c, weights, running_var, L, layer_sizes, denoising_cost):
        z_est = {}
        d_cost = []  # to store the denoising cost of all layers
        for l in range(L, -1, -1):
            print "Layer ", l, ": ", layer_sizes[l+1] if l+1 < len(layer_sizes) else None, " -> ", layer_sizes[l], ", denoising cost: ", denoising_cost[l]
            z, z_c = clean['unlabeled']['z'][l], corr['unlabeled']['z'][l]
            m, v = clean['unlabeled']['m'].get(l, 0), clean['unlabeled']['v'].get(l, 1-1e-10)
            if l == L:
                u = self.unlabeled(y_c)
            else:
                u = tf.matmul(z_est[l+1], weights['V'][l])
            u = self.batch_normalization(u)
            z_est[l] = self.g_gauss(z_c, u, layer_sizes[l])
            z_est_bn = (z_est[l] - m) / v
            # append the cost of this layer to d_cost
            d_cost.append((tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / layer_sizes[l]) * denoising_cost[l])
        # calculate total unsupervised cost by adding the denoising cost of all layers
        u_cost = tf.add_n(d_cost)
        return u_cost

    def model(self):
        self.inputs_t = tf.placeholder(tf.float32, shape=(None, self.layer_sizes_t[0]))
        self.inputs_p = tf.placeholder(tf.float32, shape=(None, self.layer_sizes_p[0]))
        self.inputs_u = tf.placeholder(tf.float32, shape=(None, self.layer_sizes_u[0]))
        self.inputs_d = tf.placeholder(tf.float32, shape=(None, self.layer_sizes_d[0]))

        w_N = tf.Variable(tf.random_normal(shape=(self.layer_sizes_p[-1] + self.layer_sizes_t[-1] + self.layer_sizes_u[-1] + self.layer_sizes_d[-1], 2), name='w_N'))
        b_N = tf.Variable(tf.random_normal(shape=(1,2)), name='b_N')
        self.outputs = tf.placeholder(tf.float32, shape=(None, 2))
        self.training = tf.placeholder(tf.bool)

        z_p = tf.Variable(tf.random_normal(shape=(self.layer_sizes_p[-1],self.layer_sizes_p[-1]), name='z_p'))
        z_t = tf.Variable(tf.random_normal(shape=(self.layer_sizes_t[-1],self.layer_sizes_t[-1]), name='z_t'))
        z_u = tf.Variable(tf.random_normal(shape=(self.layer_sizes_u[-1],self.layer_sizes_u[-1]), name='z_u'))

        h_p = tf.Variable(tf.random_normal(shape=(self.layer_sizes_p[-1],self.layer_sizes_p[-1]), name='h_p'))
        h_t = tf.Variable(tf.random_normal(shape=(self.layer_sizes_t[-1],self.layer_sizes_t[-1]), name='h_t'))
        h_u = tf.Variable(tf.random_normal(shape=(self.layer_sizes_u[-1],self.layer_sizes_u[-1]), name='h_u'))

        shapes_t = zip(self.layer_sizes_t[:-1], self.layer_sizes_t[1:])
        shapes_p = zip(self.layer_sizes_p[:-1], self.layer_sizes_p[1:])
        shapes_u = zip(self.layer_sizes_u[:-1], self.layer_sizes_u[1:])

        weights_t = dict(W=[self.wi(s, "W") for s in shapes_t], V=[self.wi(s[::-1], "V") for s in shapes_t],
                       beta=[self.bi(0.0, self.layer_sizes_t[l + 1], "beta") for l in range(self.L_t)],
                       gamma=[self.bi(1.0, self.layer_sizes_t[l + 1], "gamma") for l in range(self.L_t)])
        weights_p = dict(W=[self.wi(s, "W") for s in shapes_p], V=[self.wi(s[::-1], "V") for s in shapes_p],
                       beta=[self.bi(0.0, self.layer_sizes_p[l + 1], "beta") for l in range(self.L_p)],
                       gamma=[self.bi(1.0, self.layer_sizes_p[l + 1], "gamma") for l in range(self.L_p)])
        weights_u = dict(W=[self.wi(s, "W") for s in shapes_u], V=[self.wi(s[::-1], "V") for s in shapes_u],
                       beta=[self.bi(0.0, self.layer_sizes_u[l + 1], "beta") for l in range(self.L_u)],
                       gamma=[self.bi(1.0, self.layer_sizes_u[l + 1], "gamma") for l in range(self.L_u)])

        ewma = tf.train.ExponentialMovingAverage(decay=0.99)  # to calculate the moving averages of mean and variance
        bn_assigns = []  # this list stores the updates to be made to average mean and variance

        # average mean and variance of all layers
        running_mean_t = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in self.layer_sizes_t[1:]]
        running_var_t = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in self.layer_sizes_t[1:]]

        running_mean_p = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in self.layer_sizes_p[1:]]
        running_var_p = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in self.layer_sizes_p[1:]]

        running_mean_u = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in self.layer_sizes_u[1:]]
        running_var_u = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in self.layer_sizes_u[1:]]

        print "=== Corrupted Encoder ==="
        y_c_p, corr_p = self.encoder(self.inputs_p, self.noise_std, weights_p, self.training, ewma, running_mean_p, running_var_p, self.L_p, self.layer_sizes_p)
        y_c_t, corr_t = self.encoder(self.inputs_t, self.noise_std, weights_t, self.training, ewma, running_mean_t, running_var_t, self.L_t, self.layer_sizes_t)
        y_c_u, corr_u = self.encoder(self.inputs_u, self.noise_std, weights_u, self.training, ewma, running_mean_u, running_var_u, self.L_u, self.layer_sizes_u)

        print "=== Clean Encoder ==="
        y_p, clean_p = self.encoder(self.inputs_p, 0.0, weights_p, self.training, ewma, running_mean_p, running_var_p, self.L_p, self.layer_sizes_p)  # 0.0 -> do not add noise
        y_t, clean_t = self.encoder(self.inputs_t, 0.0, weights_t, self.training, ewma, running_mean_t, running_var_t, self.L_t, self.layer_sizes_t)  # 0.0 -> do not add noise
        y_u, clean_u = self.encoder(self.inputs_u, 0.0, weights_u, self.training, ewma, running_mean_u, running_var_u, self.L_u, self.layer_sizes_u)  # 0.0 -> do not add noise

        z_c_pt = tf.sigmoid(tf.matmul(y_c_p, z_p) + tf.matmul(y_c_t, z_t))
        z_c_pu = tf.sigmoid(tf.matmul(y_c_p, z_p) + tf.matmul(y_c_u, z_u))
        z_c_tu = tf.sigmoid(tf.matmul(y_c_t, z_t) + tf.matmul(y_c_u, z_u))

        h_c_pt = tf.tanh(tf.matmul(y_c_p, h_p) + tf.matmul(y_c_t, h_t))
        h_c_pu = tf.tanh(tf.matmul(y_c_p, h_p) + tf.matmul(y_c_u, h_u))
        h_c_tu = tf.tanh(tf.matmul(y_c_t, h_t) + tf.matmul(y_c_u, h_u))

        y_c_p_corr = (1 - z_c_tu) * h_c_tu + z_c_tu * y_c_p
        y_c_t_corr = (1 - z_c_pu) * h_c_pu + z_c_pu * y_c_t
        y_c_u_corr = (1 - z_c_pt) * h_c_pt + z_c_pt * y_c_u


        z_pt = tf.sigmoid(tf.matmul(y_p, z_p) + tf.matmul(y_t, z_t))
        z_pu = tf.sigmoid(tf.matmul(y_p, z_p) + tf.matmul(y_u, z_u))
        z_tu = tf.sigmoid(tf.matmul(y_t, z_t) + tf.matmul(y_u, z_u))

        h_pt = tf.tanh(tf.matmul(y_p, h_p) + tf.matmul(y_t, h_t))
        h_pu = tf.tanh(tf.matmul(y_p, h_p) + tf.matmul(y_u, h_u))
        h_tu = tf.tanh(tf.matmul(y_t, h_t) + tf.matmul(y_u, h_u))

        y_p_corr = (1 - z_tu) * h_tu + z_tu * y_p
        y_t_corr = (1 - z_pu) * h_pu + z_pu * y_t
        y_u_corr = (1 - z_pt) * h_pt + z_pt * y_u

        print "=== Decoder ==="

        u_cost_p = self.decoder(clean_p, corr_p, y_c_p_corr, weights_p, running_mean_p, self.L_p, self.layer_sizes_p, self.denoising_cost_p)
        u_cost_t = self.decoder(clean_t, corr_t, y_c_t_corr, weights_t, running_mean_t, self.L_t, self.layer_sizes_t, self.denoising_cost_t)
        u_cost_u = self.decoder(clean_u, corr_u, y_c_u_corr, weights_u, running_mean_u, self.L_u, self.layer_sizes_u, self.denoising_cost_u)

        y_N_p = self.labeled(y_c_p_corr)
        y_N_t = self.labeled(y_c_t_corr)
        y_N_u = self.labeled(y_c_u_corr)
        y_N_d = self.labeled(self.inputs_d)

        y_N = tf.concat([y_N_p, y_N_t, y_N_u,y_N_d], 1)
        y_N_final = tf.nn.softmax(tf.matmul(y_N, w_N) + b_N)

        cost_final = -tf.reduce_sum(tf.reduce_sum(self.outputs*tf.log(y_N_final), 1))
        loss = u_cost_t + u_cost_p + cost_final + u_cost_u
        
        self.accuracy = {}
        self.visual = {}
        prediction = tf.nn.softmax(tf.matmul(tf.concat([y_p_corr, y_t_corr, y_u_corr, self.inputs_d], 1), w_N) + b_N)
        correct_prediction_final = tf.equal(tf.argmin(prediction, 1), tf.argmin(self.outputs, 1))  # no of correct predictions
        accuracy_y_final = tf.reduce_mean(tf.cast(correct_prediction_final, "float")) * tf.constant(100.0)

        argmax_prediction = tf.argmax(prediction, 1)
        argmax_y = tf.argmax(self.outputs, 1)
        TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
        TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
        FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
        FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        self.accuracy['accuracy'] = accuracy_y_final
        self.accuracy['f1'] = f1
        self.accuracy['precision'] = precision
        self.accuracy['recall'] = recall

        self.learning_rate = tf.Variable(self.starter_learning_rate, trainable=False)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        # add the updates of batch normalization statistics to train_step
        bn_updates = tf.group(*bn_assigns)
        with tf.control_dependencies([self.train_step]):
            self.train_step = tf.group(bn_updates)


    def train(self, num_examples, data, num_epochs):
        saver = tf.train.Saver()
        print "===  Starting Session ==="
        sess = tf.Session()
        i_iter = 0
        num_iter = (num_examples/self.batch_size) * num_epochs

        ckpt = tf.train.get_checkpoint_state('checkpoints/')  # get latest checkpoint (if any)
        if ckpt and ckpt.model_checkpoint_path:
            # if checkpoint exists, restore the parameters and set epoch_n and i_iter
            saver.restore(sess, ckpt.model_checkpoint_path)
            epoch_n = int(ckpt.model_checkpoint_path.split('-')[1])
            i_iter = (epoch_n+1) * (num_examples/self.batch_size)
            print "Restored Epoch ", epoch_n
        else:
        # no checkpoint exists. create checkpoints directory if it does not exist.
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
                init = tf.global_variables_initializer()
                sess.run(init)
        init = tf.global_variables_initializer()
        sess.run(init)
        print "=== Training ==="
        print "Initial Accuracy: ", sess.run(self.accuracy, feed_dict={self.inputs_p: data.test.texts, self.inputs_t: data.test.topologys, self.inputs_u: data.test.urls, self.inputs_d:data.test.demos, self.outputs: data.test.labels, self.training: False}), "%"
        print "Initial Accuracy: ", sess.run(self.accuracy, feed_dict={self.inputs_p: data.test.texts, self.inputs_t: data.test.topologys, self.inputs_u: data.test.urls, self.inputs_d:data.test.demos, self.outputs: data.test.labels, self.training: False}), "%"

        for i in tqdm(range(i_iter, num_iter)):
            texts, topologys, urls, demos, labels = data.train.next_batch(self.batch_size)
            sess.run(self.train_step, feed_dict={self.inputs_p: texts, self.inputs_t: topologys, self.inputs_u:urls, self.inputs_d:demos, self.outputs: labels, self.training: True})
            if (i > 1) and ((i+1) % (num_iter/num_epochs) == 0):
                epoch_n = i/(num_examples/self.batch_size)
                if (epoch_n+1) >= self.decay_after:
                    # decay learning rate
                    # learning_rate = starter_learning_rate * ((num_epochs - epoch_n) / (num_epochs - decay_after))
                    ratio = 1.0 * (num_epochs - (epoch_n+1))  # epoch_n + 1 because learning rate is set for next epoch
                    ratio = max(0, ratio / (num_epochs - self.decay_after))
                    sess.run(self.learning_rate.assign(self.starter_learning_rate * ratio))
                #saver.save(sess, 'checkpoints/model.ckpt', epoch_n)
                print "Epoch ", epoch_n, ", Accuracy: ", sess.run(self.accuracy, feed_dict={self.inputs_p: data.test.texts,self.inputs_t: data.test.topologys, self.inputs_u: data.test.urls, self.inputs_d:data.test.demos,self.outputs: data.test.labels, self.training: False}), "%"
                with open('train_log', 'ab') as train_log:
                    # write test accuracy to file "train_log"
                    train_log_w = csv.writer(train_log)
                    log_i = [epoch_n] + sess.run([self.accuracy], feed_dict={self.inputs_p: data.test.texts,self.inputs_t: data.test.topologys, self.inputs_u: data.test.urls,self.inputs_d:data.test.demos, self.outputs: data.test.labels, self.training: False})
                    train_log_w.writerow(log_i)

        print "Final Accuracy: ", sess.run(self.accuracy, feed_dict={self.inputs_p: data.test.texts,self.inputs_t: data.test.topologys, self.inputs_u: data.test.urls,self.inputs_d:data.test.demos, self.outputs: data.test.labels, self.training: False}), "%"
        sess.close()
