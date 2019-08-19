from train import train_model
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Train CCSA model.')
parser.add_argument('-r', '--repetition', type=int, default=10, help='Number of repetition.')
parser.add_argument('-b', '--batchsize', type=int, default=256, help='Number of batch size in training.')
parser.add_argument('-e', '--epochs', type=int, default=80, help='Number of epoches in training.')
parser.add_argument('-a', '--alpha', type=float, default=0.25, help='Weight of contrastive loss.')
args = parser.parse_args()

if __name__ == '__main__':
    repetition = args.repetition
    batch_size = args.batchsize
    epochs = args.epochs
    alpha = args.alpha

    accs = []
    for i in range(repetition):
        acc = train_model(alpha=alpha, batch_size=batch_size, epochs=epochs)
        accs.append(acc)
        print('Repetition: {0}, Accuracy: {1}'.format(i, acc))

    m_a = np.mean(accs)

    