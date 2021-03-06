from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import TensorBoard

import os
import shutil
import numpy as np

from models import tiny_base_model, classifier
from data import sample_source_data, sample_target_data, create_pairs, load_pairs
from train_utils import plot_loss, plot_acc, printn

from keras.layers import Dense, Activation, Dropout

def eucl_dist(vects):
    eps = 1e-08
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), eps))


def eucl_dist_output_shape(shapes):
    shape1, _ = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def train_model(alpha=0.25, batch_size=32, epochs=200):
    X_s, y_s, _, _ = sample_source_data()
    X_t, y_t, X_test, y_test = sample_target_data()
    X1, X2, y1, y2, yc = create_pairs(X_s, y_s, X_t, y_t)
    #X1, X2, y1, y2, yc, X_test, y_test = load_pairs()

    base_model = tiny_base_model()
    input_s = Input(shape=(X_s.shape[1], X_s.shape[2], X_s.shape[3]))
    input_t = Input(shape=(X_t.shape[1], X_t.shape[2], X_t.shape[3]))

    processed_s = base_model(input_s)
    processed_t = base_model(input_t)

    output = classifier(processed_s)
    dist = Lambda(eucl_dist, output_shape=eucl_dist_output_shape, name='CSA')([processed_s, processed_t])

    model = Model(inputs=[input_s, input_t], outputs=[output, dist])
    model.compile(loss={'classification': 'categorical_crossentropy', 'CSA': contrastive_loss},
                  optimizer='adadelta', loss_weights={'classification': 1 - alpha, 'CSA': alpha})

    print('Model Compiled.')
    model.summary()
    if not os.path.exists('./images/'):
        os.mkdir('./images/')
    plot_model(model, to_file='./images/resnet.png')

    # Deploy TensorBoard as callbacks
    log_dir = './logs'
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    
    cb = TensorBoard(log_dir='./logs',
                    histogram_freq=0,
                    write_graph=True,
                    write_grads=True,
                    write_images=True,
                    embeddings_freq=0, 
                    embeddings_layer_names=None, 
                    embeddings_metadata=None)

    print('Start training model.')
    #history = model.fit(x=[X1, X2], y=[y1, yc], batch_size=batch_size, epochs=epochs, callbacks=[cb], validation_split=0.2)
    #plot_loss(history)

    best_acc = 0
    acc_his = [0]
    total_loss_his = []
    classification_loss_his = []
    CSA_loss_his = []
    for e in range(epochs):
        if e % 10 == 0:
            printn(str(e) + '->')
        for i in range(int(len(y2) / batch_size)):
            loss = model.train_on_batch([X1[i * batch_size:(i + 1) * batch_size, :, :, :], X2[i * batch_size:(i + 1) * batch_size, :, :, :]],
                                        [y1[i * batch_size:(i + 1) * batch_size, :], yc[i * batch_size:(i + 1) * batch_size, ]])
            loss = model.train_on_batch([X2[i * batch_size:(i + 1) * batch_size, :, :, :], X1[i * batch_size:(i + 1) * batch_size, :, :, :]],
                                        [y2[i * batch_size:(i + 1) * batch_size, :], yc[i * batch_size:(i + 1) * batch_size, ]])


        out = model.predict([X_test, X_test])
        acc_v = np.argmax(out[0], axis=1) - np.argmax(y_test, axis=1)
        acc = (len(acc_v) - np.count_nonzero(acc_v) + .0000001) / len(acc_v)

        if best_acc < acc:
            best_acc = acc

        acc_his.append(acc)
        total_loss_his.append(loss[2])
        classification_loss_his.append(loss[1])
        CSA_loss_his.append(loss[0])
    """
    if not os.path.exists('./model/'):
        os.mkdir('./model/')
    model.save('./model/CCSA.h5')
    """

    plot_loss(total_loss_his, classification_loss_his, CSA_loss_his)
    plot_acc(acc_his)

    return best_acc