package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	ImageDirArg = 1
	OutNetArg   = 2

	StepSize       = 1e-4
	BatchSize      = 12
	ValidationSize = 0.1

	LogInterval = 4
)

func main() {
	rand.Seed(time.Now().UnixNano())

	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "image_dir out_net")
		os.Exit(1)
	}

	log.Println("Loading samples...")
	samples, err := imagenet.NewSampleSet(os.Args[ImageDirArg])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read sample listing:", err)
		os.Exit(1)
	}
	validation, training := sgd.HashSplit(samples, ValidationSize)
	log.Println("Loaded", validation.Len(), "validation,", training.Len(), "training.")

	log.Println("Loading/creating network...")
	network, err := LoadOrCreateNetwork(os.Args[OutNetArg], samples)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to create network:", err)
		os.Exit(1)
	}
	gradienter := &sgd.Momentum{
		Gradienter: &neuralnet.BatchRGradienter{
			Learner:       network.BatchLearner(),
			CostFunc:      &neuralnet.DotCost{},
			MaxBatchSize:  BatchSize,
			MaxGoroutines: 1,
		},
		Momentum: 0.9,
	}

	log.Println("Training...")
	var miniBatch int
	var lastBatch sgd.SampleSet
	sgd.SGDMini(gradienter, training, StepSize, BatchSize, func(s sgd.SampleSet) bool {
		if miniBatch%LogInterval == 0 {
			validationCost := randomSubsetCost(validation, network)
			newCost := randomSubsetCost(s, network)
			if lastBatch == nil {
				log.Printf("batch=%d validation=%f training=%f", miniBatch,
					validationCost, newCost)
			} else {
				log.Printf("batch=%d validation=%f training=%f last=%f", miniBatch,
					validationCost, newCost, randomSubsetCost(lastBatch, network))
			}
			lastBatch = s.Copy()
		}
		miniBatch++
		return true
	})

	serialized, err := network.Serialize()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to serialize network:", err)
		os.Exit(1)
	}
	if err := ioutil.WriteFile(os.Args[OutNetArg], serialized, 0755); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save network:", err)
		os.Exit(1)
	}
}

func randomSubsetCost(s sgd.SampleSet, n neuralnet.Network) float64 {
	if s.Len() > BatchSize {
		s = s.Copy()
		sgd.ShuffleSampleSet(s)
		s = s.Subset(0, BatchSize)
	}
	var inVec linalg.Vector
	var outVec linalg.Vector
	for i := 0; i < s.Len(); i++ {
		sample := s.GetSample(i).(neuralnet.VectorSample)
		inVec = append(inVec, sample.Input...)
		outVec = append(outVec, sample.Output...)
	}
	out := n.BatchLearner().Batch(&autofunc.Variable{Vector: inVec}, s.Len())
	return neuralnet.DotCost{}.Cost(outVec, out).Output()[0]
}
