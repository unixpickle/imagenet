package main

import (
	"io/ioutil"
	"math"
	"math/rand"
	"os"

	"github.com/unixpickle/weakai/neuralnet"
)

const (
	MeanSampleCount = 100
)

var ConvFilterCounts = []int{48, 64, 128, 128, 128, 128}
var PoolingLayers = map[int]bool{1: true, 2: true, 3: true, 4: true, 5: true}
var HiddenSizes = []int{2048, 2048}

func LoadOrCreateNetwork(path string, samples SampleSet) (neuralnet.Network, error) {
	data, err := ioutil.ReadFile(path)
	if err == nil {
		return neuralnet.DeserializeNetwork(data)
	} else if !os.IsNotExist(err) {
		return nil, err
	}

	mean, variance := pixelStats(samples)
	net := neuralnet.Network{
		&neuralnet.RescaleLayer{
			Bias:  -mean,
			Scale: 1 / math.Sqrt(variance),
		},
	}

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

func pixelStats(samples SampleSet) (mean, variance float64) {
	var count int
	for i := 0; i < MeanSampleCount; i++ {
		sampleIdx := rand.Intn(samples.Len())
		sample := samples.GetSample(sampleIdx).(neuralnet.VectorSample)
		for _, brightness := range sample.Input {
			mean += brightness
			variance += brightness * brightness
			count++
		}
	}
	mean /= float64(count)
	variance /= float64(count)
	variance -= mean * mean
	return
}
