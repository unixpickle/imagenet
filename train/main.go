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

	StepSize  = 1e-2
	BatchSize = 30

	ValidationSize   = 0.1
	ValidationSubset = 50
	SubTrainingSize  = 250
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
	gradienter := &sgd.Adam{
		Gradienter: &neuralnet.BatchRGradienter{
			Learner:       network.BatchLearner(),
			CostFunc:      &neuralnet.DotCost{},
			MaxBatchSize:  2,
			MaxGoroutines: 2,
		},
	}

	sgd.ShuffleSampleSet(samples)
	validation := samples.Subset(0, int(float64(samples.Len())*ValidationSize))
	training := samples.Subset(validation.Len(), samples.Len())

	subTraining := training.Copy().Subset(0, SubTrainingSize).(SampleSet)

	var epoch int
	log.Println("Training...")
	sgd.SGDInteractive(gradienter, subTraining, StepSize, BatchSize, func() bool {
		log.Printf("Epoch: %d; subset cost: %f", epoch, randomSubsetCost(validation, network))
		s := training.Copy()
		sgd.ShuffleSampleSet(s)
		s = s.Subset(0, SubTrainingSize)
		copy(subTraining, s.(SampleSet))
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
	if s.Len() > ValidationSubset {
		s = s.Copy()
		sgd.ShuffleSampleSet(s)
		s = s.Subset(0, ValidationSubset)
	}
	return neuralnet.TotalCost(&neuralnet.DotCost{}, n, s)
}
