// Command post_train produces an *imagenet.Classifier for
// a neural network.
// As part of doing this, it converts batch normalization
// layers into affine transforms.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/anynet/anyconv"
	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/serializer"
)

func main() {
	var imgDir string
	var inNet string
	var outNet string

	var batchSize int
	var sampleCount int

	flag.StringVar(&imgDir, "samples", "", "sample directory")
	flag.StringVar(&inNet, "in", "", "input network")
	flag.StringVar(&outNet, "out", "", "output network")
	flag.IntVar(&batchSize, "batch", 8, "evaluation batch size")
	flag.IntVar(&sampleCount, "total", 512, "total samples for BatchNorm replacement")

	flag.Parse()

	if imgDir == "" || inNet == "" || outNet == "" {
		fmt.Fprintln(os.Stderr, "Required flags: -in, -out, and -samples")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
		os.Exit(1)
	}

	log.Println("Loading samples...")
	samples, err := imagenet.NewSampleList(imgDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read sample listing:", err)
		os.Exit(1)
	}
	rand.Seed(time.Now().UnixNano())
	anysgd.Shuffle(samples)
	if sampleCount < samples.Len() {
		samples = samples.Slice(0, sampleCount).(imagenet.SampleList)
	}

	log.Println("Loading network...")
	var cl *imagenet.Classifier
	if err = serializer.LoadAny(inNet, &cl); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read network:", err)
		os.Exit(1)
	}

	log.Println("Replacing BatchNorm layers...")
	var numReplaced int
	pt := &anyconv.PostTrainer{
		Samples:   samples,
		Fetcher:   &anyff.Trainer{},
		BatchSize: batchSize,
		Net:       cl.Net,
		StatusFunc: func(bn *anyconv.BatchNorm) {
			numReplaced++
			log.Println("Replaced", numReplaced, "BatchNorms.")
		},
	}
	if err = pt.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "Post-training error:", err)
		os.Exit(1)
	}

	log.Println("Saving classifier...")
	if err = serializer.SaveAny(outNet, cl); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save:", err)
		os.Exit(1)
	}
}
