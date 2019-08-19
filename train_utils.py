import matplotlib.pyplot as plt
import sys

def plot_acc(history):  
    #plt.plot(history.history['acc'], 'r', label='train_acc')
    plt.plot(range(len(history)), history, 'r', label='val_acc')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig('images/acc.png')
    plt.close('all')


def plot_loss(total_loss_his, classification_loss_his, CSA_loss_his):
    plt.plot(range(1, len(classification_loss_his)+1), classification_loss_his, 'r', label='Classification_loss')
    plt.plot(range(1, len(CSA_loss_his)+1), CSA_loss_his, 'g', label='CSA_loss')
    plt.plot(range(1, len(total_loss_his)+1), total_loss_his, 'b', label='Total_loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig('images/loss.png')
    plt.close('all')

def printn(string):
    sys.stdout.write(string)
    sys.stdout.flush()
