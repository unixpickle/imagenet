package main

import (
	"io/ioutil"
	"sort"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/serializer"
)

func LoadOrCreateClassifier(path, modelPath, samplePath string) (*imagenet.Classifier, error) {
	var net anynet.Net
	if err := serializer.LoadAny(path, &net); err == nil {
		return turnIntoClassifier(net, samplePath)
	}

	var cl *imagenet.Classifier
	if err := serializer.LoadAny(path, &cl); err == nil {
		return cl, nil
	}

	modelData, err := ioutil.ReadFile(modelPath)
	if err != nil {
		return nil, err
	}
	res, err := anyconv.FromMarkup(anyvec32.CurrentCreator(), string(modelData))
	if err != nil {
		return nil, err
	}
	return turnIntoClassifier(res.(anynet.Net), samplePath)
}

func turnIntoClassifier(net anynet.Net, samplePath string) (*imagenet.Classifier, error) {
	listing, err := ioutil.ReadDir(samplePath)
	if err != nil {
		return nil, err
	}
	var dirNames []string
	for _, item := range listing {
		if item.IsDir() {
			dirNames = append(dirNames, item.Name())
		}
	}
	sort.Strings(dirNames)
	return &imagenet.Classifier{
		InWidth:  imagenet.InputImageSize,
		InHeight: imagenet.InputImageSize,
		Net:      net,
		Classes:  dirNames,
	}, nil
}
