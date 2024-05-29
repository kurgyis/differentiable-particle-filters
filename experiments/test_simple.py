import tensorflow as tf

import local_env 
from methods.dpf import DPF
from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, remove_state
from utils.exp_utils import get_default_hyperparams
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')


def train_dpf(task='nav01', data_path='../data/100s', model_path='../models/tmp', plot=False):

    # load training data and add noise
    train_data = load_data(data_path=data_path, filename=task + '_train')
    noisy_train_data = noisyfy_data(train_data)

    # reset tensorflow graph
    tf.reset_default_graph()

    # instantiate method
    hyperparams = get_default_hyperparams()
    method = DPF(**hyperparams['global'])

    with tf.Session() as session:
        # train method and save result in model_path
        method.fit(session, noisy_train_data, model_path, **hyperparams['train'], plot_task=task, plot=plot)


def test_dpf(task='nav01', data_path='../data/100s', model_path='../models/tmp'):

    # load test data
    test_data = load_data(data_path=data_path, filename=task + '_test')
    noisy_test_data = noisyfy_data(test_data)
    test_batch_iterator = make_batch_iterator(noisy_test_data, seq_len=50)

    # reset tensorflow graph
    tf.reset_default_graph()

    # instantiate method
    hyperparams = get_default_hyperparams()
    method = DPF(**hyperparams['no_propose'])

    with tf.Session() as session:
        # load method and apply to new data
        method.load(session, model_path)
        for i in range(10):
            test_batch = next(test_batch_iterator)
            test_batch_input = remove_state(test_batch, provide_initial_state=False)
            result = method.predict(session, test_batch_input, **hyperparams['test'])

    return result


if __name__ == '__main__':
    pred_states, particle_list, probability_list, intermediate_states  = test_dpf()
    print(pred_states.shape)
    print(particle_list.shape)
    print(probability_list.shape)
    print(intermediate_states.shape)
    plt.plot(pred_states[:, 0], pred_states[:, 1], 'r')
