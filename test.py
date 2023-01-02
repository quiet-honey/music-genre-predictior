import numpy as np
import Layer
import matplotlib.pyplot as plt
import optimizer


### data 불러오기 ###
print("start data")
from data_preprocessing import data_preprocessing
(x_train, t_train), (x_test, t_test) = data_preprocessing()
print("end data")
print("====================================================")
### 신경망 생성 ###
# input_size:   feature 수
# input_size:   one-hot-encoding label 수
# hidden_size_list: 각 은닉층의 퍼셉트론(노드) 수
# weight_init_std:  가중치 초기화 표준편차
# use_bn:   배치 정규화 사용 여부
# use_do:   드롭 아웃 사용 여부
network = Layer.MultiLayer(input_size=x_train.shape[1], 
                           output_size=t_train.shape[1], 
                           hidden_size_list=[100, 100, 100, 100, 100], 
                           weight_init_std=0.01, 
                           use_bn=True, 
                           use_do=True)

### 학습 변수 설정 ###
# epochs_num:   학습 epoch 수
# train_size:   학습 data의 개수
# batch_size:   배치 크기
# lr:           학습률
# iter_per_epoch: epoch당 반복 횟수
epochs_num = 50
train_size = x_train.shape[0]
batch_size = 256
iter_per_epoch = int(train_size / batch_size)
lr = 1e-3

### 결과 출력을 위한 list선언 ###
# train_loss_list:  학습 시 loss값 저장 list
# train_acc_list:   학습 시 정확도 저장 list
# train_loss_list:  평가 시 loss값 저장 list
train_loss_list = []
train_acc_list = []
test_acc_list = []

### optimizer 선언 ###
# optimizer는 Adam으로 선언 및 학습률 전달
optim = optimizer.Adam(lr)

### 초기 loss값 저장 ###
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
loss = network.loss(x_batch, t_batch)
train_loss_list.append(loss)


train_acc = network.accuracy(x_train, t_train, train_flg=False)
test_acc = network.accuracy(x_test, t_test, train_flg=False)
train_acc_list.append(train_acc)
test_acc_list.append(test_acc)

### 학습 ###
for i in range(epochs_num):
    print("Epoch {}/{}".format(i+1, epochs_num))
    print("[", end='')
    for j in range(iter_per_epoch + 1):
        if j%3==0:
            print(">", end='')
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        # 신경망의 순전파, 역전파를 모두 진행하는 grad함수 호출하여 각layer의 기울기를 반환받는다
        grad = network.gradient(x_batch, t_batch, train_flg=True)
        # 반환된 grad 변수와 신경망의 param변수(W1, W2, b1, b2 etc...)들을 함께 optimizer에 전달하여 학습
        optim.step(network.params, grad)
    print(']')

    # 새로운 batch를 랜덤으로 선정
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 해당 data로 loss값 계산 및 저장
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # test, training의 정확도를 계산 후  저장
    train_acc = network.accuracy(x_train, t_train, train_flg=False)
    test_acc = network.accuracy(x_test, t_test, train_flg=False)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    print("train acc: {} %".format(train_acc*100))
    print("test acc:  {} %".format(test_acc*100))
    print("loss:  {}".format(loss))
        
plt.plot(train_acc_list, 'bo-')
plt.xlim([0, 51])
plt.plot(test_acc_list, 'm^-')
plt.ylim([0,1])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

plt.plot(train_loss_list, 'rx--')
plt.show()