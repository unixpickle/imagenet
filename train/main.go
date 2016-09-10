package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	ImageDirArg = 1
	OutNetArg   = 2

	StepSize            = 1e-5
	BatchNormStabilizer = 1e-3
	BatchSize           = 12
	ValidationSize      = 0.1

	MaxCache = BatchSize * 64 * 64 * 128
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "image_dir out_net")
		os.Exit(1)
	}

	log.Println("Loading samples...")
	samples, err := NewSampleSet(os.Args[ImageDirArg])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read sample listing:", err)
		os.Exit(1)
	}

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

	sgd.ShuffleSampleSet(samples)
	validation := samples.Subset(0, int(float64(samples.Len())*ValidationSize))
	training := samples.Subset(validation.Len(), samples.Len())

	log.Println("Training...")
	var miniBatch int
	var lastBatch sgd.SampleSet
	sgd.SGDMini(gradienter, training, StepSize, BatchSize, func(s sgd.SampleSet) bool {
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
	return neuralnet.TotalCost(&neuralnet.DotCost{}, n, s)
}
