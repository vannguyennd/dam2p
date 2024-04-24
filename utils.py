import tensorflow as tf
import numpy as np
from random import shuffle
import random
from collections import Counter
import scipy.io as io


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    np.random.seed(123)
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def get_data_values(p_data, num_input_vocabulary):
    sum_data = []

    for ib_data in range(p_data.shape[0]):
        sub_data = p_data[ib_data]
        batch_data = []
        for if_data in range(sub_data.shape[0]):
            data = np.zeros(num_input_vocabulary)
            data[sub_data[if_data]] = 1
            batch_data.append(data)

        sum_data.append(np.array(batch_data))

    return np.stack(sum_data)


def get_data_values_advance(p_data, num_input_vocabulary):
    sum_data = []

    for ib_data in range(p_data.shape[0]):
        sub_data = p_data[ib_data]
        batch_data = []
        for if_data in range(sub_data.shape[0]):
            data = np.zeros(num_input_vocabulary)
            oc_data = Counter(sub_data[if_data])
            for idx in range(data.shape[0]):
                if oc_data[idx] != 0:
                    data[idx] = oc_data[idx]
            batch_data.append(data)

        sum_data.append(np.array(batch_data))

    return np.stack(sum_data)


def get_batch(p_inputs, p_outputs, batch_size):
    data_size = p_inputs.shape[0]
    sample_values = np.random.choice(data_size, batch_size, replace=True)
    return p_inputs[sample_values], p_outputs[sample_values]


def compute_acc_pn(l_lb_prediction, source_test_labels, batch_size):
    s_p_acc = 0
    c_p_acc = 0
    s_n_acc = 0
    c_n_acc = 0

    for idx in range(batch_size):
        if source_test_labels[idx] == 0:
            s_p_acc += 1
            if l_lb_prediction[idx] == 0:
                c_p_acc += 1
        else:
            s_n_acc += 1
            if l_lb_prediction[idx] == 1:
                c_n_acc += 1

    return c_p_acc, s_p_acc, c_n_acc, s_n_acc


def convert_to_int(source_test_labels):
    result_values = []

    for i_values in source_test_labels:
        result_values.append(int(i_values))

    return np.array(result_values)


def compute_metric(l_lb_prediction, source_test_labels, batch_size):
    tp, fn, fp, tn = 0, 0, 0, 0

    for idx in range(batch_size):
        if source_test_labels[idx] == 0:
            if l_lb_prediction[idx] == 0:
                tn += 1
        elif source_test_labels[idx] == 1:
            if l_lb_prediction[idx] == 1:
                tp += 1
            else:
                fn += 1
        else:
            print('error!')

    for i_dx in range(batch_size):
        if l_lb_prediction[i_dx] == 1:
            if source_test_labels[i_dx] != 1:
                fp += 1

    return tp, fp, tn, fn
