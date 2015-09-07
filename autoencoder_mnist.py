
import os
import numpy as np
import matplotlib.pyplot as plot

from ann import ANN


NOP = 0
TEST = 1
STOP = 2


def check_cmd():
    if os.path.exists("TEST"):
        os.remove("TEST")
        return TEST

    return NOP


def print_num(img, mul=1):
    m = (img.reshape((28,28)) * mul).astype(np.uint8) 

    m[(m>0) & (m<=150)] = 1
    m[(m>0) & (m> 150)] = 5

    print m




def test(ann, images):

    N = images.shape[0]
    ii = range(N)
    np.random.shuffle(ii)
    ii = ii[:10]

    to_show = None
    for i in ii:
        tmp = images[i].copy()
	p = ann.predict_proba(tmp)

	tmp = tmp.reshape((28,28))
	p   = p.reshape((28,28))

        print_num(p, 255)

        tmp *= 255
        p   *= 255


        if to_show == None:
            to_show = np.concatenate((tmp, p), axis=1)
        else:
            to_show = np.concatenate(( to_show, np.concatenate((tmp, p), axis=1) ), axis=0)
        
    to_show = to_show.astype(np.uint8)

    plot.clf()
    plot.imshow(to_show)
    plot.show()



N = 60000
S = 28*28

path = "./data/train-images-idx3-ubyte"

images = np.fromfile(path, dtype=np.uint8, sep='')[16:]
images = images.reshape((N, S))




#tmp = np.concatenate((tmp, tmp2), axis=1)
#
#plot.imshow(tmp)
#plot.show()




# prepare data
images = images.astype(np.float64)
#images[images > 0] = 255
images /= 255

print_num(images[1], 255)


train_ii = range(N)
np.random.shuffle(train_ii)

N_train = int(N * .8)
N_test  = N - N_train
test_ii = train_ii[N_train:]
train_ii = train_ii[:N_train]





M = 128
ann = ANN([S, M, S], .0)


total_cnt = 0
total_avr_cost = 0.


avr_cost = 0.
cnt = 0.
EPOCHES = 1000
for e in range(EPOCHES):
    np.random.shuffle(train_ii)

    for i in train_ii:
        tmp = images[i]

        ww, bb = ann.get_weights()

        ann.partial_fit(tmp, tmp, n_iter=1)
        if ann.cost.value > 1000:
            ann.set_weights(ww, bb)
            print "BAD cost", ann.cost.value, "idx", i
        else:
            avr_cost += ann.cost.value
            cnt += 1.

            total_avr_cost += ann.cost.value
            total_cnt += 1

            if cnt == 200:
                print "avr cost", (avr_cost / cnt), "processed", total_cnt, "total avr cost", (total_avr_cost / total_cnt)
                cnt = 0.
                avr_cost = 0.


        cmd = check_cmd()
        if TEST == cmd:
            i = np.random.randint(0, N - 10)
            test(ann, images[test_ii])



test(ann, images[test_ii])




