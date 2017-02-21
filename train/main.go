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
	"github.com/unixpickle/essentials"
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
	var weightDecay float64
	var logInterval int
	var modelFile string

	flag.StringVar(&imageDir, "samples", "", "sample directory")
	flag.StringVar(&outNet, "out", "out_net", "network file")
	flag.Float64Var(&stepSize, "step", 0.001, "step size")
	flag.IntVar(&batchSize, "batch", 12, "batch size")
	flag.Float64Var(&validationSize, "validation", 0.1, "validation fraction")
	flag.Float64Var(&weightDecay, "decay", 1e-4, "L2 weight decay")
	flag.IntVar(&logInterval, "logint", 4, "validation log interval")
	flag.StringVar(&modelFile, "model", "models/orig.txt", "model markup file")

	flag.Parse()

	if imageDir == "" || outNet == "" {
		fmt.Fprintln(os.Stderr, "Required flags: -samples and -out")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
		os.Exit(1)
	}

	log.Println("Loading/creating network...")
	classifier, err := LoadOrCreateClassifier(outNet, modelFile, imageDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to create network:", err)
		os.Exit(1)
	}
	network := classifier.Net

	paramCount := 0
	for _, p := range network.Parameters() {
		paramCount += p.Vector.Len()
	}
	log.Println("Network has", paramCount, "parameters.")

	log.Println("Loading samples...")
	samples, err := imagenet.NewSampleList(imageDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read sample listing:", err)
		os.Exit(1)
	}
	validation, training := anysgd.HashSplit(samples, validationSize)
	log.Println("Loaded", validation.Len(), "validation,", training.Len(), "training.")

	t := &anyff.Trainer{
		Net: network,
		Cost: &anynet.L2Reg{
			Penalty: weightDecay,
			Params:  network.Parameters(),
			Wrapped: anynet.DotCost{},
		},
		Params:  network.Parameters(),
		Average: true,
	}

	vBatches := make(chan anysgd.Batch, 1)
	go func() {
		for {
			anysgd.Shuffle(validation)
			valid := validation.Slice(0, batchSize)
			batch, err := t.Fetch(valid)
			if err != nil {
				essentials.Die(err)
			}
			vBatches <- batch
		}
	}()

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
				batch := <-vBatches
				vCost := anyvec.Sum(t.TotalCost(batch.(*anyff.Batch)).Output())
				log.Printf("iter %d: cost=%v validation=%v", iterNum, vCost, t.LastCost)
			}
			iterNum++
		},
		BatchSize: batchSize,
	}

	log.Println("Press ctrl+c once to stop...")
	err = s.Run(rip.NewRIP().Chan())
	if err != nil {
		fmt.Fprintln(os.Stderr, "Training error:", err)
	}

	if err := serializer.SaveAny(outNet, classifier); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save network:", err)
		os.Exit(1)
	}
}
