import numpy as np
import matplotlib.pyplot as plt

train_hist = np.load('./results/ResNet18_train_20200220-123106.npy')
test_hist = np.load('./results/ResNet18_test_20200220-123106.npy')
train_loss, train_acc = train_hist[:,0], train_hist[:,1]
test_loss, test_acc = test_hist[:,0], test_hist[:,1]

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(train_acc, label = 'train')
ax1.plot(test_acc, label = 'test')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2 = fig.add_subplot(122)
ax2.plot(train_loss, label = 'train')
ax2.plot(test_loss, label = 'test')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

fig.tight_layout()

plt.show()