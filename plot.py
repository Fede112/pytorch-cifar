import numpy as np
import matplotlib.pyplot as plt

normal_hist_train = np.load('./results/ResNet18Siamese_train_20200303-211028.npy')
seed_hist_train = np.load('./results/ResNet18Siamese_train_20200304-122122.npy')
normal_loss_train, normal_acc_train = normal_hist_train[:,0], normal_hist_train[:,1]
seed_loss_train, seed_acc_train = seed_hist_train[:,0], seed_hist_train[:,1]


normal_hist_test = np.load('./results/ResNet18Siamese_test_20200303-211028.npy')
seed_hist_test = np.load('./results/ResNet18Siamese_test_20200304-122122.npy')
normal_loss_test, normal_acc_test = normal_hist_test[:,0], normal_hist_test[:,1]
seed_loss_test, seed_acc_test = seed_hist_test[:,0], seed_hist_test[:,1]



fig = plt.figure()
ax1 = fig.add_subplot(121)
# train
ax1.plot(normal_acc_test, label = 'normal_test')
ax1.plot(seed_acc_test, label = 'seeding_test')
# test
# ax1.plot(normal_acc_test, label = 'normal_test')
# ax1.plot(seed_acc_test, label = 'seeding_test')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2 = fig.add_subplot(122)
# train
ax2.plot(normal_loss_train, label = 'normal_train')
ax2.plot(seed_loss_train, label = 'seeding_train')
# test
# ax2.plot(normal_loss_test, label = 'normal_test')
# ax2.plot(seed_loss_test, label = 'seeding_test')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()

fig.tight_layout()

plt.show()