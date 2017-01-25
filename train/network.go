package main

import (
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/serializer"
)

const (
	MeanSampleCount = 100
)

var ConvFilterCounts = []int{48, 64, 96, 128, 128, 128}
var PoolingLayers = map[int]bool{0: true, 1: true, 2: true, 3: true, 4: true, 5: true}
var HiddenSizes = []int{2048, 2048}

func LoadOrCreateNetwork(path string, samples imagenet.SampleList) (anynet.Net, error) {
	var net anynet.Net
	if err := serializer.LoadAny(path, &net); err == nil {
		return net, nil
	}

	c := anyvec32.CurrentCreator()

	layerSize := imagenet.InputImageSize
	layerDepth := 3

	for i, filterCount := range ConvFilterCounts {
		layer := &anyconv.Conv{
			FilterWidth:  3,
			FilterHeight: 3,
			FilterCount:  filterCount,
			StrideX:      1,
			StrideY:      1,
			InputWidth:   layerSize,
			InputHeight:  layerSize,
			InputDepth:   layerDepth,
		}
		layer.InitRand(c)
		layerDepth = filterCount
		layerSize = layer.OutputWidth()
		net = append(net, layer)
		net = append(net, anyconv.NewBatchNorm(c, filterCount))
		if PoolingLayers[i] {
			poolLayer := &anyconv.MaxPool{
				SpanX:       2,
				SpanY:       2,
				InputWidth:  layerSize,
				InputHeight: layerSize,
				InputDepth:  layerDepth,
			}
			net = append(net, poolLayer)
			layerSize = poolLayer.OutputWidth()
		}
		net = append(net, anynet.ReLU)
	}

	inputVecSize := layerSize * layerSize * layerDepth
	for _, hiddenSize := range HiddenSizes {
		net = append(net, anynet.NewFC(c, inputVecSize, hiddenSize))
		net = append(net, anynet.ReLU)
		inputVecSize = hiddenSize
	}
	net = append(net, anynet.NewFC(c, inputVecSize, samples.ClassCount()))
	net = append(net, anynet.LogSoftmax)

	return net, nil
}
