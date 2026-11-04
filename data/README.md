# Image data for the training runs

`fl_mlp2.py`, `fl_mlp3.py` and `fl_mlp_5k.py` read MNIST and Fashion-MNIST in the original IDX format:

```
data/fmnist/train-images-idx3-ubyte.gz  train-labels-idx1-ubyte.gz  t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz
data/mnist/  (same four files)
```
Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist (MIT). MNIST: http://yann.lecun.com/exdb/mnist/ or the
mirror at https://ossci-datasets.s3.amazonaws.com/mnist/. Override the locations with `P3_DATA` and `P3_DATA_MNIST`.

`rev_round14_cnn.py` reads CIFAR-10 from `data/cifar10/cifar10_jpeg_mirror.npz` (arrays `X_train`, `y_train`, `X_test`,
`y_test`, `classes`; uint8 images 32x32x3), built from the JPEG re-encoding of CIFAR-10 at
https://github.com/YoongiKim/CIFAR-10-images (the official archive at https://www.cs.toronto.edu/~kriz/cifar.html was not
reachable from the build machine); override the folder with `P3_DATA_CIFAR`. The conversion is in `src/make_cifar_npz.py`.
