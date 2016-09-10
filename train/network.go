package main

import (
	"io/ioutil"
	"os"

	"github.com/unixpickle/batchnorm"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	MeanSampleCount = 100
)

var ConvFilterCounts = []int{48, 64, 96, 128, 128, 128}
var PoolingLayers = map[int]bool{0: true, 1: true, 2: true, 3: true, 4: true, 5: true}
var HiddenSizes = []int{2048, 2048}

func LoadOrCreateNetwork(path string, samples SampleSet) (neuralnet.Network, error) {
	data, err := ioutil.ReadFile(path)
	if err == nil {
		return neuralnet.DeserializeNetwork(data)
	} else if !os.IsNotExist(err) {
		return nil, err
	}

	net := neuralnet.Network{}

	layerSize := InputImageSize
	layerDepth := 3

	for i, filterCount := range ConvFilterCounts {
		layer := &neuralnet.ConvLayer{
			FilterWidth:  3,
			FilterHeight: 3,
			FilterCount:  filterCount,
			Stride:       1,
			InputWidth:   layerSize,
			InputHeight:  layerSize,
			InputDepth:   layerDepth,
		}
		layerDepth = filterCount
		layerSize = layer.OutputWidth()
		net = append(net, layer)
		net = append(net, batchnorm.NewLayer(filterCount))
		if PoolingLayers[i] {
			poolLayer := &neuralnet.MaxPoolingLayer{
				XSpan:       2,
				YSpan:       2,
				InputWidth:  layerSize,
				InputHeight: layerSize,
				InputDepth:  layerDepth,
			}
			net = append(net, poolLayer)
			layerSize = poolLayer.OutputWidth()
		}
		net = append(net, &neuralnet.ReLU{})
	}

	inputVecSize := layerSize * layerSize * layerDepth
	for _, hiddenSize := range HiddenSizes {
		net = append(net, &neuralnet.DenseLayer{
			InputCount:  inputVecSize,
			OutputCount: hiddenSize,
		})
		net = append(net, &neuralnet.ReLU{})
		inputVecSize = hiddenSize
	}
	net = append(net, &neuralnet.DenseLayer{
		InputCount:  inputVecSize,
		OutputCount: samples.ClassCount(),
	})
	net = append(net, &neuralnet.LogSoftmaxLayer{})

	net.Randomize()

	return net, nil
}
