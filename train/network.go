package main

import (
	"io/ioutil"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/serializer"
)

const (
	MeanSampleCount = 100
)

var ConvFilterCounts = []int{48, 64, 96, 128, 128, 128}
var PoolingLayers = map[int]bool{0: true, 1: true, 2: true, 3: true, 4: true, 5: true}
var HiddenSizes = []int{2048, 2048}

func LoadOrCreateNetwork(path, modelPath string) (anynet.Net, error) {
	var net anynet.Net
	if err := serializer.LoadAny(path, &net); err == nil {
		return net, nil
	}

	modelData, err := ioutil.ReadFile(modelPath)
	if err != nil {
		return nil, err
	}
	res, err := anyconv.FromMarkup(anyvec32.CurrentCreator(), string(modelData))
	if err != nil {
		return nil, err
	}
	return res.(anynet.Net), nil
}
