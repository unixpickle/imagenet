package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/pkg/profile"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
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

	defer profile.Start(profile.MemProfile).Stop()

	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "image_dir out_net")
		os.Exit(1)
	}

	log.Println("Loading samples...")
	samples, err := imagenet.NewSampleList(os.Args[ImageDirArg])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read sample listing:", err)
		os.Exit(1)
	}
	validation, training := anysgd.HashSplit(samples, ValidationSize)
	log.Println("Loaded", validation.Len(), "validation,", training.Len(), "training.")

	log.Println("Loading/creating network...")
	network, err := LoadOrCreateNetwork(os.Args[OutNetArg], samples)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to create network:", err)
		os.Exit(1)
	}

	t := &anyff.Trainer{
		Net:     network,
		Cost:    anynet.DotCost{},
		Params:  network.Parameters(),
		Average: true,
	}

	var iterNum int
	s := &anysgd.SGD{
		Fetcher:     t,
		Gradienter:  t,
		Transformer: &anysgd.Adam{},
		Samples:     samples,
		Rater:       anysgd.ConstRater(0.001),
		StatusFunc: func(b anysgd.Batch) {
			if iterNum%LogInterval == 0 {
				log.Printf("iter %d: cost=%v", iterNum, t.LastCost)
			} else {
				anysgd.Shuffle(validation)
				valid := validation.Slice(0, BatchSize)
				batch, _ := t.Fetch(valid)
				vCost := anyvec.Sum(t.TotalCost(batch.(*anyff.Batch)).Output())
				log.Printf("iter %d: cost=%v validation=%v", iterNum, vCost, t.LastCost)
			}
			iterNum++
		},
		BatchSize: 100,
	}

	log.Println("Press ctrl+c once to stop...")
	s.Run(rip.NewRIP().Chan())

	if err := serializer.SaveAny(os.Args[OutNetArg], network); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save network:", err)
		os.Exit(1)
	}
}
