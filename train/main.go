package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	var imageDir string
	var outNet string
	var stepSize float64
	var batchSize int
	var validationSize float64
	var logInterval int

	flag.StringVar(&imageDir, "samples", "", "sample directory")
	flag.StringVar(&outNet, "out", "out_net", "network file")
	flag.Float64Var(&stepSize, "step", 0.001, "step size")
	flag.IntVar(&batchSize, "batch", 12, "batch size")
	flag.Float64Var(&validationSize, "validation", 0.1, "validation fraction")
	flag.IntVar(&logInterval, "logint", 4, "validation log interval")

	flag.Parse()

	if imageDir == "" || outNet == "" {
		fmt.Fprintln(os.Stderr, "Required flags: -samples and -out")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
		os.Exit(1)
	}

	log.Println("Loading samples...")
	samples, err := imagenet.NewSampleList(imageDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read sample listing:", err)
		os.Exit(1)
	}
	validation, training := anysgd.HashSplit(samples, validationSize)
	log.Println("Loaded", validation.Len(), "validation,", training.Len(), "training.")

	log.Println("Loading/creating network...")
	network, err := LoadOrCreateNetwork(outNet, samples)
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
		Rater:       anysgd.ConstRater(stepSize),
		StatusFunc: func(b anysgd.Batch) {
			if iterNum%logInterval != 1 {
				log.Printf("iter %d: cost=%v", iterNum, t.LastCost)
			} else {
				anysgd.Shuffle(validation)
				valid := validation.Slice(0, batchSize)
				batch, _ := t.Fetch(valid)
				vCost := anyvec.Sum(t.TotalCost(batch.(*anyff.Batch)).Output())
				log.Printf("iter %d: cost=%v validation=%v", iterNum, vCost, t.LastCost)
			}
			iterNum++
		},
		BatchSize: batchSize,
	}

	log.Println("Press ctrl+c once to stop...")
	s.Run(rip.NewRIP().Chan())

	if err := serializer.SaveAny(outNet, network); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save network:", err)
		os.Exit(1)
	}
}
