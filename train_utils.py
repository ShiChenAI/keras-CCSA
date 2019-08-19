import matplotlib.pyplot as plt

def plot_acc(history):  
    plt.plot(history.history['acc'], 'r', label='train_acc')
    plt.plot(history.history['val_acc'], 'b', label='val_acc')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig('images/acc.png')
    plt.close('all')


def plot_loss(history):
    plt.plot(history.history['loss'], 'r', label='train_loss')
    plt.plot(history.history['classification_loss'], 'g', label='classification_loss')
    plt.plot(history.history['CSA_loss'], 'b', label='CSA_loss')
    plt.plot(history.history['val_loss'], 'k', label='val_loss')
    plt.plot(history.history['val_classification_loss'], 'c', label='val_classification_loss')
    plt.plot(history.history['val_CSA_loss'], 'm', label='val_CSA_loss')
    
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig('images/loss.png')
    plt.close('all')