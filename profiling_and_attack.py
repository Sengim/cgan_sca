import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from scipy.stats import entropy
from scipy.stats import multivariate_normal
from utils import *


def mlp(classes, number_of_samples, learning_rate=0.001):
    input_shape = (number_of_samples)
    input_layer = Input(shape=input_shape, name="input_layer")

    x = Dense(100, kernel_initializer="glorot_normal", activation="elu")(input_layer)
    x = Dense(100, kernel_initializer="glorot_normal", activation="elu")(x)
    x = Dense(100, kernel_initializer="glorot_normal", activation="elu")(x)
    x = Dense(100, kernel_initializer="glorot_normal", activation="elu")(x)

    output_layer = Dense(classes, activation='softmax', name=f'output')(x)

    m_model = Model(input_layer, output_layer, name='mlp_softmax')
    optimizer = Adam(learning_rate=learning_rate)
    m_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    m_model.summary()
    return m_model


def guessing_entropy(predictions, labels_guess, good_key, key_rank_attack_traces, key_rank_report_interval=1):
    """
    Function to compute Guessing Entropy
    - this function computes a list of key candidates, ordered by their probability of being the correct key
    - if this function returns final_ge=1, it means that the correct key is actually indicated as the most likely one.
    - if this function returns final_ge=256, it means that the correct key is actually indicated as the least likely one.
    - if this function returns final_ge close to 128, it means that the attack is wrong and the model is simply returing a random key.

    :return
    - final_ge: the guessing entropy of the correct key
    - guessing_entropy: a vector indicating the value 'final_ge' with respect to the number of processed attack measurements
    - number_of_measurements_for_ge_1: the number of processed attack measurements necessary to reach final_ge = 1
    """

    nt = len(predictions)

    key_rank_executions = 40

    # key_ranking_sum = np.zeros(key_rank_attack_traces)
    key_ranking_sum = np.zeros(
        int(key_rank_attack_traces / key_rank_report_interval))

    predictions = np.log(predictions + 1e-36)

    probabilities_kg_all_traces = np.zeros((nt, 256))
    for index in range(nt):
        probabilities_kg_all_traces[index] = predictions[index][
            np.asarray([int(leakage[index])
                        for leakage in labels_guess[:]])
        ]

    for run in range(key_rank_executions):
        r = np.random.choice(
            range(nt), key_rank_attack_traces, replace=False)
        probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
        key_probabilities = np.zeros(256)

        kr_count = 0
        for index in range(key_rank_attack_traces):

            key_probabilities += probabilities_kg_all_traces_shuffled[index]
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]

            if (index + 1) % key_rank_report_interval == 0:
                key_ranking_good_key = list(
                    key_probabilities_sorted).index(good_key) + 1
                key_ranking_sum[kr_count] += key_ranking_good_key
                kr_count += 1

    guessing_entropy = key_ranking_sum / key_rank_executions

    number_of_measurements_for_ge_1 = key_rank_attack_traces
    if guessing_entropy[int(key_rank_attack_traces / key_rank_report_interval) - 1] < 2:
        for index in range(int(key_rank_attack_traces / key_rank_report_interval) - 1, -1, -1):
            if guessing_entropy[index] > 2:
                number_of_measurements_for_ge_1 = (
                                                          index + 1) * key_rank_report_interval
                break

    final_ge = guessing_entropy[int(
        key_rank_attack_traces / key_rank_report_interval) - 1]
    print("GE = {}".format(final_ge))
    print("Number of traces to reach GE = 1: {}".format(
        number_of_measurements_for_ge_1))

    return final_ge, guessing_entropy, number_of_measurements_for_ge_1


def information(model, labels, num_classes):
    """
    implements I(K;L)

    p_k = the distribution of the sensitive variable K
    data = the samples we 'measured'. It its the n^k_p samples from p(l|k)
    model = the estimated model \hat{p}(l|k).

    returns an estimated of mutual information
    """

    labels = np.array(labels, dtype=np.uint8)
    p_k = np.ones(num_classes, dtype=np.float64)
    for k in range(num_classes):
        p_k[k] = np.count_nonzero(labels == k)
    p_k /= len(labels)

    acc = entropy(p_k, base=2)  # we initialize the value with H(K)

    y_pred = np.array(model + 1e-36)

    for k in range(num_classes):
        trace_index_with_label_k = np.where(labels == k)[0]
        y_pred_k = y_pred[trace_index_with_label_k, k]

        y_pred_k = np.array(y_pred_k)
        if len(y_pred_k) > 0:
            p_k_l = np.sum(np.log2(y_pred_k)) / len(y_pred_k)
            acc += p_k[k] * p_k_l

    print(f"PI: {acc}")

    return acc


def template_training(X, Y, pool=False):
    num_clusters = max(Y) + 1
    classes = np.unique(Y)
    # assign traces to clusters based on lables
    HW_catagory_for_traces = [[] for _ in range(num_clusters)]
    for i in range(len(X)):
        HW = Y[i]
        HW_catagory_for_traces[HW].append(X[i])
    HW_catagory_for_traces = [np.array(HW_catagory_for_traces[HW]) for HW in range(num_clusters)]

    # calculate Covariance Matrices
    # step 1: calculate mean matrix of POIs
    meanMatrix = np.zeros((num_clusters, len(X[0])))
    for i in range(num_clusters):
        meanMatrix[i] = np.mean(HW_catagory_for_traces[i], axis=0)
    # step 2: calculate covariance matrix
    covMatrix = np.zeros((num_clusters, len(X[0]), len(X[0])))
    for HW in range(num_clusters):
        for i in range(len(X[0])):
            for j in range(len(X[0])):
                x = HW_catagory_for_traces[HW][:, i]
                y = HW_catagory_for_traces[HW][:, j]
                covMatrix[HW, i, j] = np.cov(x, y)[0][1]
    if pool:
        covMatrix[:] = np.mean(covMatrix, axis=0)
    return meanMatrix, covMatrix, classes


# Calculate probability of the most possible cluster for each traces
def template_attacking(meanMatrix, covMatrix, X_test, template_classes, labels, num_classes):
    labels = np.array(labels, dtype=np.uint8)
    p_k = np.ones(num_classes, dtype=np.float64)
    for k in range(num_classes):
        p_k[k] = np.count_nonzero(labels == k)
    p_k /= len(labels)

    number_traces = X_test.shape[0]
    proba = np.zeros((number_traces, template_classes.shape[0]))
    rv_array = []
    m = 1e-6
    for idx in range(len(template_classes)):
        rv_array.append(multivariate_normal(meanMatrix[idx], covMatrix[idx], allow_singular=True))

    for i in range(number_traces):
        if (i % 2000 == 0):
            print(str(i) + '/' + str(number_traces))
        proba[i] = [o.pdf(X_test[i]) for o in rv_array]
        proba[i] = np.multiply(proba[i], p_k) / np.sum(np.multiply(proba[i], p_k))

    return proba


def attack(dataset, generator, features_dim: int, attack_model=None, synthetic_traces=True, original_traces=False, batch_size=400):
    """ Generate a batch of synthetic measurements with the trained generator """
    if original_traces:
        features_target_attack = np.array(dataset.dataset_target.x_attack)
        features_target_profiling = np.array(dataset.dataset_target.x_profiling)
    else:
        if synthetic_traces:
            features_target_attack = np.array(generator.predict([dataset.dataset_target.x_attack]))
            features_target_profiling = np.array(generator.predict([dataset.dataset_target.x_profiling]))
        else:
            features_target_attack = np.array(dataset.features_target_attack)
            features_target_profiling = np.array(dataset.features_target_profiling)

    """ Define a neural network (MLP) to be trained with synthetic traces """

    if attack_model is None:
        model = mlp(dataset.dataset_target.classes, features_dim)
    else:
        model = attack_model
    model.fit(
        x=features_target_profiling,
        y=to_categorical(dataset.dataset_target.profiling_labels, num_classes=dataset.dataset_target.classes),
        batch_size=batch_size,
        verbose=2,
        epochs=20,
        shuffle=True,
        validation_data=(
            features_target_attack, to_categorical(dataset.dataset_target.attack_labels, num_classes=dataset.dataset_target.classes)),
        callbacks=[])

    """ Predict the trained MLP with target/attack measurements """
    predictions = model.predict(features_target_attack)
    """ Check if we are able to recover the key from the target/attack measurements """
    ge, ge_vector, nt = guessing_entropy(predictions, dataset.dataset_target.labels_key_hypothesis_attack,
                                         dataset.dataset_target.correct_key_attack, 2000)
    pi = information(predictions, dataset.dataset_target.attack_labels, dataset.dataset_target.classes)
    return ge, nt, pi, ge_vector


def template_attack(dataset):
    features_target_profiling, features_target_attack = get_lda_features(dataset.dataset_target,
                                                                     dataset.target_byte_target,
                                                                     n_poi=10)
    """ Template Attack """
    mean_v, cov_v, template_classes = template_training(features_target_profiling, dataset.dataset_target.profiling_labels, pool=False)
    predictions = template_attacking(mean_v, cov_v, features_target_attack, template_classes, dataset.dataset_target.attack_labels,
                                     dataset.dataset_target.classes)

    """ Check if we are able to recover the key from the target/attack measurements """
    ge, ge_vector, nt = guessing_entropy(predictions, dataset.dataset_target.labels_key_hypothesis_attack,
                                         dataset.dataset_target.correct_key_attack, 2000)
    pi = information(predictions, dataset.dataset_target.attack_labels, dataset.dataset_target.classes)
    return ge, nt, pi, ge_vector
