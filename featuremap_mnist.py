
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


def print_num(img, img_size, mul=1):
    m = (img.reshape((img_size,img_size)) * mul).astype(np.uint8) 

    m[(m>0) & (m<=150)] = 1
    m[(m>0) & (m> 150)] = 5

    print m




def test(ann, images, img_size, S, F, neurons_num):

    ww, bb = ann.get_weights()
    ss = ann.ss[: len(ann.ss)/2 + 1]
    flt = ANN(ss, .0)
    ww_size = 0
    bb_size = 0
    for l in range(1, len(ss)):
        ww_size += ss[l-1] * ss[l]
        bb_size += ss[l]
    flt.set_weights(ww[:ww_size], bb[:bb_size])


    N = images.shape[0]
    ii = range(N)
    np.random.shuffle(ii)
    ii = ii[:10]

    to_show = None
    for i in ii:
        tmp = images[i].copy()
        tmp = tmp.reshape((img_size,img_size))


        flt_img = np.zeros((neurons_num,neurons_num))

        r = 0
        c = 0
        for patch in patches(tmp, img_size, S, F):
            p = flt.predict_proba(patch.flatten())

            flt_img[r,c] = p[0,1]
            c = (c + 1) % neurons_num
            if c == 0:
                r = (r + 1) % neurons_num


        tmp = tmp.reshape((img_size,img_size))

        tmp *= 255
        flt_img *= 255

	flt_img = np.concatenate((flt_img, [[0]*(img_size-neurons_num)]* neurons_num),           axis=1)
        flt_img = np.concatenate((flt_img, [[0]* img_size             ]*(img_size-neurons_num)), axis=0)


        if to_show == None:
            to_show = np.concatenate((tmp, flt_img), axis=1)
        else:
            to_show = np.concatenate(( to_show, np.concatenate((tmp, flt_img), axis=1) ), axis=0)
        
    to_show = to_show.astype(np.uint8)

    plot.clf()
    plot.imshow(to_show)
    plot.show()



def patches(img, img_size, S, F):
    img = img.reshape((img_size,img_size))
    for r in range(0, img_size-F, S):
        for c in range(0, img_size-F, S):
            ret = img[r:r+F, c:c+F]
            yield ret



def feed_patches(img, img_size, S, F, ann):
    avr_cost = 0.
    cnt = 0.
    for patch in patches(img, img_size, S, F):
        ann.partial_fit(patch.flatten(), patch.flatten(), A=.0001)
        avr_cost += ann.cost.value
        cnt += 1.

    return avr_cost / cnt




N = 60000
S = 28*28
IMG_SIZE = 28

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

print_num(images[1], IMG_SIZE, 255)


train_ii = range(N)
np.random.shuffle(train_ii)

N_train = int(N * .8)
N_test  = N - N_train
test_ii = train_ii[N_train:]
train_ii = train_ii[:N_train]



# NOTE: for one side (dimention) only
S = 2        # stride
F = 8        # patch size
neurons_num = (IMG_SIZE - F) / S + 1


PS = F * F
M  = 1

ann = ANN([PS, 32, 8, M, 8, 32, PS], .0)


total_cnt = 0
total_avr_cost = 0.


avr_cost = 0.
cnt = 0.
EPOCHES = 1000
for e in range(EPOCHES):
    np.random.shuffle(train_ii)

    for i in train_ii:
        tmp = images[i]

        cost = feed_patches(tmp, IMG_SIZE, S, F, ann)

        avr_cost += cost
        cnt += 1.

        total_avr_cost += cost
        total_cnt += 1

        if cnt == 200:
            print "avr cost", (avr_cost / cnt), "processed", total_cnt, "total avr cost", (total_avr_cost / total_cnt)
            cnt = 0.
            avr_cost = 0.


        cmd = check_cmd()
        if TEST == cmd:
            i = np.random.randint(0, N - 10)
            test(ann, images[test_ii], IMG_SIZE, S, F, neurons_num)


test(ann, images[test_ii], IMG_SIZE, S, F, neurons_num)




