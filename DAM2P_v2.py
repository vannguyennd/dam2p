from tensorflow.contrib import rnn
from flip_gradient import flip_gradient
from utils import *
import time
import os
from sklearn import metrics as mt
import argparse


opt_type = 'adam'
cell_type = 'lstm'
num_layers = 1
ti_steps = 35
input_vocab = 4025
st_len = 25
max_gradient_norm = 1.0
steps = 10000


class DomainModel(object):
    """DAM2P domain adaptation model."""

    def __init__(self, source_vul_num, args):
        self.gamma_init = 1.0 / args.dnn
        self.train_gamma = False
        self.args = args
        self.source_vul_num = source_vul_num
        self._build_model()

    def _build_model(self):
        tf.reset_default_graph()

        self.learning_rate = tf.placeholder(tf.float32, [])
        self.d_rate = tf.placeholder(tf.float32, [])
        self.t_rate = tf.placeholder(tf.float32, [])

        self.X_source_target = tf.placeholder(tf.float32, shape=(None, ti_steps, input_vocab))
        self.Y_source_target = tf.placeholder(tf.int32, shape=[None])
        self.Y_domain = tf.placeholder(tf.int32, shape=[None])

        self.ll = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])

        self.weights_st = weight_variable([2 * self.args.hnn * ti_steps, self.args.dnn])
        self.biases_st = bias_variable([self.args.dnn])

        with tf.name_scope('source_target_generator'):
            x_st = tf.unstack(self.X_source_target, ti_steps, 1)

            if cell_type == 'lstm':
                if num_layers > 1:
                    fw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(self.args.hnn) for _ in range(num_layers)])
                    bw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.LSTMCell(self.args.hnn) for _ in range(num_layers)])
                else:
                    fw_cell_st = rnn.LSTMCell(self.args.hnn)
                    bw_cell_st = rnn.LSTMCell(self.args.hnn)
            elif cell_type == 'gru':
                if num_layers > 1:
                    fw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.GRUCell(self.args.hnn) for _ in range(num_layers)])
                    bw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.GRUCell(self.args.hnn) for _ in range(num_layers)])
                else:
                    fw_cell_st = rnn.GRUCell(self.args.hnn)
                    bw_cell_st = rnn.GRUCell(self.args.hnn)
            else:
                if num_layers > 1:
                    fw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.BasicRNNCell(self.args.hnn) for _ in range(num_layers)])
                    bw_cell_st = tf.contrib.rnn.MultiRNNCell([rnn.BasicRNNCell(self.args.hnn) for _ in range(num_layers)])
                else:
                    fw_cell_st = rnn.BasicRNNCell(self.args.hnn)
                    bw_cell_st = rnn.BasicRNNCell(self.args.hnn)

            l_outputs_st, _, _ = rnn.static_bidirectional_rnn(fw_cell_st, bw_cell_st, x_st, dtype=tf.float32, scope='fw_cell_st')
            l_outputs_st = tf.transpose(tf.stack(l_outputs_st, axis=0), perm=[1, 0, 2])
            l_outputs_st = tf.reshape(l_outputs_st, [-1, 2 * self.args.hnn * ti_steps])

            outputs_st = tf.matmul(l_outputs_st, self.weights_st) + self.biases_st
            lo_gits_st = tf.reshape(outputs_st, [-1, self.args.dnn])

            self.features = lo_gits_st

        with tf.name_scope("cross-kernel-classifier"):
            self.input_of_ckl = self.features

            log_gamma = tf.get_variable(name='log_gamma', shape=[1],
                                        initializer=tf.constant_initializer(np.log(self.gamma_init)),
                                        trainable=self.train_gamma)

            random_weight = tf.get_variable(name="unit_noise", shape=[self.input_of_ckl.get_shape()[1],
                                                                      self.args.rfe//2],
                                            initializer=tf.random_normal_initializer(), trainable=False)

            omega = tf.multiply(tf.exp(log_gamma), random_weight, name='omega')
            omega_x = tf.matmul(self.input_of_ckl, omega)
            phi_x_tilde = tf.concat([tf.cos(omega_x), tf.sin(omega_x)], axis=1, name='phi_x_tilde')

            self.w_rf = tf.Variable(tf.truncated_normal((self.args.rfe, 1), stddev=0.1), name='w_rf')
            self.rho = tf.Variable(0.0, dtype=tf.float32, name='rho')

            self.l2_regularization = 0.5 * tf.reduce_sum(tf.square(self.w_rf))

        with tf.name_scope("loss_svm"):
            self.Y_svm = self.Y_source_target
            self.Y_svm = tf.subtract(tf.cast(self.Y_svm, tf.float32), 0.5)
            self.Y_svm = tf.reshape(tf.sign(self.Y_svm), shape=(self.args.bs, 1))

            y_svm_source_vul = tf.slice(self.Y_svm, [0, 0], [int(self.source_vul_num), -1])
            y_svm_source_non = tf.slice(self.Y_svm, [int(self.source_vul_num), 0],
                                        [int(self.args.bs/2) - int(self.source_vul_num), -1])

            phi_x_tilde_source_vul = tf.slice(phi_x_tilde, [0, 0], [int(self.source_vul_num), -1])
            phi_x_tilde_source_non = tf.slice(phi_x_tilde, [int(self.source_vul_num), 0],
                                              [int(self.args.bs/2) - int(self.source_vul_num), -1])

            phi_x_tilde_target = tf.slice(phi_x_tilde, [int(self.args.bs/2), 0], [-1, -1])

            w_phi_minus_rho_source_vul = 0 - tf.multiply(y_svm_source_vul,
                                                         (tf.matmul(phi_x_tilde_source_vul, self.w_rf) - self.rho))

            w_phi_minus_rho_source_non = 1 - tf.multiply(y_svm_source_non,
                                                         (tf.matmul(phi_x_tilde_source_non, self.w_rf) - self.rho))

            w_phi_minus_rho_target = self.rho - tf.matmul(phi_x_tilde_target, self.w_rf)

        with tf.name_scope('domain_discriminator'):
            feat = flip_gradient(self.features, self.ll)

            w_d_0 = weight_variable([self.args.dnn, self.args.dnn])
            b_d_0 = bias_variable([self.args.dnn])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, w_d_0) + b_d_0)

            w_d_1 = weight_variable([self.args.dnn, self.args.dnn])
            b_d_1 = bias_variable([self.args.dnn])
            d_h_fc1 = tf.nn.relu(tf.matmul(d_h_fc0, w_d_1) + b_d_1)

            w_d_2 = weight_variable([self.args.dnn, 2])
            b_d_2 = bias_variable([2])
            lo_gits_d = tf.matmul(d_h_fc1, w_d_2) + b_d_2

            self.domain_prediction = tf.nn.softmax(lo_gits_d)
            self.loss_op_d = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lo_gits_d, labels=self.Y_domain)

        with tf.name_scope("train_and_predict"):
            self.loss_source_vul = tf.reduce_mean(tf.maximum(0.0, w_phi_minus_rho_source_vul))
            self.loss_source_non = tf.reduce_mean(tf.maximum(0.0, w_phi_minus_rho_source_non))
            self.loss_target = tf.reduce_mean(tf.maximum(0.0, w_phi_minus_rho_target))

            self.loss_mmp = self.l2_regularization + self.loss_source_vul + self.loss_source_non + self.args.lam * self.loss_target
            domain_loss = tf.reduce_mean(self.loss_op_d)
            self.total_loss = self.loss_mmp + self.args.alp * domain_loss

            parameters = tf.trainable_variables()
            gradients = tf.gradients(self.total_loss, parameters)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

            if opt_type == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            elif opt_type == 'grad':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            else:
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

            self.dan_train_op = optimizer.apply_gradients(zip(clipped_gradients, parameters))

        with tf.name_scope("predict"):
            self.lb_prediction = tf.sign(tf.matmul(phi_x_tilde, self.w_rf) - self.rho)
            correct_domain_prediction = tf.nn.in_top_k(self.domain_prediction, self.Y_domain, 1)
            self.domain_acc = tf.reduce_mean(tf.cast(correct_domain_prediction, tf.float32))

            with tf.name_scope("init_save"):
                self.init = tf.global_variables_initializer()
                self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=150)


source_data = np.load('./datasets/peg_source_data.npy', allow_pickle=True)
source_train_non, source_train_non_labels = source_data.item()['train_data_non'], source_data.item()['train_label_non']
source_train_vul, source_train_vul_labels = source_data.item()['train_data_vul'], source_data.item()['train_label_vul']
source_test, source_test_labels = source_data.item()['data_test'], source_data.item()['label_test']

target_data = np.load('./datasets/png_target_data.npy', allow_pickle=True)
target_train, target_train_labels = target_data.item()['data_train'], target_data.item()['label_train']
target_test, target_test_labels = target_data.item()['data_test'], target_data.item()['label_test']


today = time.strftime('%Y%m%d')
hour = time.strftime('%h%s')


def train_and_evaluate(training_mode, args, verbose=True):
    """helper to run the model with different training modes."""

    saved_dir = "./save_results_nos/" + str(today) + "_" + str(hour) + '/' + str(args.alp) \
            + '_' + str(args.hnn) + '_' + str(args.lam) + '_' + str(args.rfe) + '/'

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    result_file = open('./save_results_nos_v2/' + str(today) + "_" + str(hour) + '/' + 'peg_png.txt', 'a+')
    save_high_values = []

    vul_each_batch = 5
    non_each_batch = int(args.bs/2) - vul_each_batch

    model = DomainModel(vul_each_batch, args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        check_point = tf.train.get_checkpoint_state(saved_dir)
        if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
            print("load model parameters from %s" % check_point.model_checkpoint_path)
            model.saver.restore(sess, check_point.model_checkpoint_path)
        else:
            print("create the model with fresh parameters")
            sess.run(model.init)

        gen_source_non_batch = batch_generator([source_train_non, source_train_non_labels], non_each_batch)
        gen_source_vul_batch = batch_generator([source_train_vul, source_train_vul_labels], vul_each_batch)

        gen_target_batch = batch_generator([target_train, target_train_labels], int(args.bs / 2))

        source_test_values = get_data_values(source_test, input_vocab)
        gen_source_test_batch = batch_generator([source_test_values, source_test_labels], int(args.bs))
        s_size = int(source_test.shape[0] // args.bs)

        target_test_values = get_data_values(target_test, input_vocab)
        gen_target_test_batch = batch_generator([target_test_values, target_test_labels], int(args.bs))
        t_size = int(target_test.shape[0] // args.bs)

        y_domain_labels = np.concatenate(([np.tile([0], [int(args.bs / 2)]), np.tile([1], [int(args.bs / 2)])]))

        print('p_d_rate: ' + str(args.alp) + '; ' + 'p_num_hidden: ' +
              str(args.hnn) + '; ' + 'p_t_rate: ' + str(args.lam) + '; ' + 'p_num_features: ' + str(args.rfe))

        result_file.write('p_d_rate: ' + str(args.alp) + '; ' + 'p_num_hidden: ' +
                          str(args.hnn) + '; ' + 'p_t_rate: ' + str(args.lam) + '; '
                          + 'p_num_features: ' + str(args.rfe) + '\n')

        for i_step in range(args.ts):
            p = float(i_step) / steps
            ll = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1
            l_lr = args.lr
            lr = l_lr / (1. + 10 * p) ** 0.75

            if training_mode == 'dan':
                x0_non, y0_non = gen_source_non_batch.__next__()
                x0_vul, y0_vul = gen_source_vul_batch.__next__()
                x1, y1 = gen_target_batch.__next__()

                x0_non = get_data_values(x0_non, input_vocab)
                x0_vul = get_data_values(x0_vul, input_vocab)
                x_1 = get_data_values(x1, input_vocab)

                x = np.concatenate((x0_vul, x0_non, x_1), axis=0)
                y = np.concatenate((y0_vul, y0_non, y1), axis=0)

                _, batch_loss, d_acc = \
                    sess.run([model.dan_train_op, model.total_loss, model.domain_acc],
                             feed_dict={model.X_source_target: x, model.Y_source_target: y,
                                        model.Y_domain: y_domain_labels, model.train: True, model.ll: ll,
                                        model.learning_rate: lr, model.d_rate: args.alp,
                                        model.t_rate: args.lam})

                if verbose and i_step % 1 == 0:
                    print('epoch: ' + str(i_step))
                    result_file.write('epoch: ' + str(i_step) + '\n')
                    print('loss: %f  domain_acc: %f' % (batch_loss, d_acc))
                    result_file.write('loss: %f  domain_acc: %f \n' % (batch_loss, d_acc))

                    full_y_predict_train = np.array([])
                    full_y_target_train = np.array([])

                    for _ in range(s_size):
                        batch_x_s, batch_y_s = gen_source_test_batch.__next__()
                        batch_y_s = convert_to_int(batch_y_s)
                        full_y_target_train = np.append(full_y_target_train, batch_y_s)
                        s_l_lb_prediction = sess.run(model.lb_prediction, feed_dict={model.X_source_target: batch_x_s,
                                                                                     model.Y_source_target: batch_y_s,
                                                                                     model.train: False})
                        s_l_lb_prediction[s_l_lb_prediction == -1] = 0
                        full_y_predict_train = np.append(full_y_predict_train, s_l_lb_prediction)

                    src_test_acc = mt.accuracy_score(y_true=full_y_target_train, y_pred=full_y_predict_train)
                    src_test_pre = mt.precision_score(y_true=full_y_target_train, y_pred=full_y_predict_train)
                    src_test_f1 = mt.f1_score(y_true=full_y_target_train, y_pred=full_y_predict_train)
                    src_test_re = mt.recall_score(y_true=full_y_target_train, y_pred=full_y_predict_train)
                    src_test_auc = mt.roc_auc_score(y_true=full_y_target_train, y_score=full_y_predict_train)
                    tn, fp, fn, tp = mt.confusion_matrix(y_true=full_y_target_train,
                                                        y_pred=full_y_predict_train).ravel()
                    if (fp + tn) == 0:
                        fpr = -1.0
                    else:
                        fpr = float(fp) / (fp + tn)

                    if (tp + fn) == 0:
                        fnr = -1.0
                    else:
                        fnr = float(fn) / (tp + fn)

                    print('src_acc: %.5f ; src_pre: %.5f ; src_f1: %.5f ; src_re: %.5f ; '
                        'src_auc: %.5f' % (src_test_acc, src_test_pre, src_test_f1, src_test_re, src_test_auc))

                    result_file.write("fpr: %.5f; " % fpr)
                    result_file.write("fnr: %.5f; " % fnr)
                    result_file.write("src_test_acc: %.5f; " % src_test_acc)
                    result_file.write("src_test_pre: %.5f; " % src_test_pre)
                    result_file.write("src_test_f1: %.5f; " % src_test_f1)
                    result_file.write("src_test_re: %.5f; " % src_test_re)
                    result_file.write("src_test_auc: %.5f \n" % src_test_auc)

                    full_y_predict_train = np.array([])
                    full_y_target_train = np.array([])
                    for _ in range(t_size):
                        batch_x, batch_y = gen_target_test_batch.__next__()
                        batch_y = convert_to_int(batch_y)
                        full_y_target_train = np.append(full_y_target_train, batch_y)
                        t_l_lb_prediction = sess.run(model.lb_prediction, feed_dict={model.X_source_target: batch_x,
                                                                                     model.Y_source_target: batch_y,
                                                                                     model.train: False})
                        t_l_lb_prediction[t_l_lb_prediction == -1] = 0
                        full_y_predict_train = np.append(full_y_predict_train, t_l_lb_prediction)

                    trg_test_acc = mt.accuracy_score(y_true=full_y_target_train, y_pred=full_y_predict_train)
                    trg_test_pre = mt.precision_score(y_true=full_y_target_train, y_pred=full_y_predict_train)
                    trg_test_f1 = mt.f1_score(y_true=full_y_target_train, y_pred=full_y_predict_train)
                    trg_test_re = mt.recall_score(y_true=full_y_target_train, y_pred=full_y_predict_train)
                    trg_test_auc = mt.roc_auc_score(y_true=full_y_target_train, y_score=full_y_predict_train)
                    tn, fp, fn, tp = mt.confusion_matrix(y_true=full_y_target_train,
                                                         y_pred=full_y_predict_train).ravel()
                    if (fp + tn) == 0:
                        fpr = -1.0
                    else:
                        fpr = float(fp) / (fp + tn)

                    if (tp + fn) == 0:
                        fnr = -1.0
                    else:
                        fnr = float(fn) / (tp + fn)

                    print('trg_acc: %.5f ; trg_pre: %.5f ; trg_f1: %.5f ; trg_re: %.5f ; '
                          'trg_auc: %.5f' % (trg_test_acc, trg_test_pre, trg_test_f1, trg_test_re, trg_test_auc))

                    result_file.write("fpr: %.5f; " % fpr)
                    result_file.write("fnr: %.5f; " % fnr)
                    result_file.write("trg_test_acc: %.5f; " % trg_test_acc)
                    result_file.write("trg_test_pre: %.5f; " % trg_test_pre)
                    result_file.write("trg_test_f1: %.5f; " % trg_test_f1)
                    result_file.write("trg_test_re: %.5f; " % trg_test_re)
                    result_file.write("trg_test_auc: %.5f \n" % trg_test_auc)

                    if trg_test_f1 >= 0.8:
                        save_high_values.append(trg_test_f1)
    return save_high_values


def main():
    """
    ===for the training process===
    for example,
    python DAM2P_v2.py --lr=1e-3 --alp=1e-1 --lam=1e-3 --hnn=128 --rfe=512 --ts=150
    Note: please refer to our paper to the whole list of hyper-parameters ranges used.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=1e-3, type=float, help="The learning rate used in the training process.")
    parser.add_argument("--alp", default=1e-1, type=float, help="The trade-off hyper-parameter for the domain adaptation loss.")
    parser.add_argument("--lam", default=1e-3, type=float, help="The trade-off hyper-parameter for the target domain loss.")
    parser.add_argument("--hnn", default=128, type=int, help="The number of neurons in the hidden layer of used in LSTMs.")
    parser.add_argument("--dnn", default=300, type=int, help="The number of neurons in the hidden layer of used in deep feedforward layers.")
    parser.add_argument("--rfe", default=512, type=int, help="The number of random features used in the random feature map.")
    parser.add_argument("--bs", default=100, type=int, help="The batch size of data samples.")
    parser.add_argument("--ts", default=150, type=int, help="The number of steps used to train the model.")

    args = parser.parse_args()

    print('DAM2P training and evaluation')
    save_high_values = train_and_evaluate('dan', args, verbose=True)
    print(save_high_values)


if __name__ == "__main__":
    main()