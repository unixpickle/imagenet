# imagenet

[ImageNet](http://image-net.org) is a massive collection of categorized real-world images. It is an indispensable resource for various machine learning tasks.

I will use this repository for all of the tools I build around ImageNet, including fetching tools, image recognition tools, etc.

# Preliminaries

All of the tools in this repository use the [Go programming language](https://golang.org/doc/install). Make sure that you have Go installed and have a [GOPATH](https://golang.org/doc/code.html#GOPATH) configured.

Once you have Go, you can download all of the dependencies for this repository as follows:

```
$ go get -u -d github.com/unixpickle/imagenet/...
```

The above command will also download this repository itself. You can `cd` into the repository like so (on Linux or macOS):

```
$ cd $GOPATH/src/github.com/unixpickle/imagenet
```

# Pre-trained models

If you do not want to train your own ImageNet classifier, you can download a pre-trained classifier.

 * [ResNet-34](http://aqnichol.com/networks/resnet34)
 * [ResNet-34 (with BatchNorms)](http://aqnichol.com/networks/resnet34_batchnorm)

# Fetching

To download ImageNet images, you can use the [fetch](fetch) tool. You should create a directory where you would like to save the downloaded images (e.g. `/path/to/images`). You should have a text file with one line per WordNet ID (WNID), such as [wnids/ilsvrc.txt](wnids/ilsvrc.txt). You should also decide the maximum number of images you'd like to download for each WNID. If you want to download everything, you can set this to a large number (e.g. 100000).

To commence a download, run the command like so:

```
$ cd $GOPATH/src/github.com/unixpickle/imagenet/fetch
$ go run *.go /path/to/wnids.txt 100000 /path/to/images
```

The download may take several hours.

# Training

To train a classifier on ImageNet images, use the [train](train) tool. You will likely want to use the GPU, meaning that you should follow the instructions [here](https://godoc.org/github.com/unixpickle/cuda#hdr-Building) on setting up CUDA with Go. You can run the train command as follows:

```
$ cd $GOPATH/src/github.com/unixpickle/imagenet/train
$ go run *.go -samples /path/to/images \
  -out /path/to/output/file \
  -model models/resnet_34.txt \
  -batch 32 \
  -step 0.1 \
  -momentum 0.9 \
  -bigbatch 8
```

All of those arguments can be tuned. In that example, they are configured to match the [ResNet](https://arxiv.org/abs/1512.03385) paper. You can use the `-help` flag for more usage information.

To gracefully pause training, press ctrl+c exactly once (pressing it multiple times terminates without saving). You will likely want to pause training several times to lower the learning rate. However, it is recommended that you pause as infrequently as possible, since the samples are reshuffled whenever you resume (so the sample distribution will become uneven).

# Post-training

After training is complete, the [BatchNorm](https://arxiv.org/pdf/1502.03167.pdf) layers in the model need to be replaced with true averages. To do this, use the [post_train](post_train) tool:

```
$ cd $GOPATH/src/github.com/unixpickle/imagenet/post_train
$ go run *.go -samples /path/to/images \
  -total 4096 \
  -in /path/to/trained/model \
  -out /path/to/finalized/model
```

The `-total` argument specifies how many samples to run through the network for the rolling average. A larger number results in a more accurate model, but post-training will take longer.
