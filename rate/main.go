package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"sort"
	"time"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet/anysgd"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/serializer"

	_ "github.com/unixpickle/batchnorm"
)

func main() {
	var classifierPath string
	var sampleDir string
	var topN int

	flag.StringVar(&classifierPath, "classifier", "", "classifier file")
	flag.StringVar(&sampleDir, "samples", "", "sample directory")
	flag.IntVar(&topN, "topn", 1, "top N rating")

	flag.Parse()

	if classifierPath == "" || sampleDir == "" {
		fmt.Fprintln(os.Stderr, "Required flags: -classifier and -samples")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
		os.Exit(1)
	}

	log.Println("Loading classifier...")
	var classifier *imagenet.Classifier
	if err := serializer.LoadAny(classifierPath, &classifier); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load classifier:", err)
		os.Exit(1)
	}

	log.Println("Loading samples...")
	samples, err := imagenet.NewSampleList(sampleDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load samples:", err)
		os.Exit(1)
	}

	rand.Seed(time.Now().UnixNano())
	anysgd.Shuffle(samples)

	sampleChan := make(chan *imagenet.Sample, 1)
	go func() {
		for i := 0; i < samples.Len(); i++ {
			sampleChan <- &samples[i]
		}
		close(sampleChan)
	}()

	outChan := make(chan bool)

	go func() {
		rateSamples(topN, classifier, sampleChan, outChan)
		close(outChan)
	}()

	printResults(outChan)
}

func rateSamples(n int, c *imagenet.Classifier, samples <-chan *imagenet.Sample,
	out chan<- bool) {
	for sample := range samples {
		ins := imagenet.TestingImages(sample.Path)
		var outSum anyvec.Vector
		for _, x := range ins {
			res := c.Net.Apply(anydiff.NewConst(x), 1).Output()
			if outSum == nil {
				outSum = res.Copy()
			} else {
				outSum.Add(res)
			}
		}

		topClasses := sortByIndex(outSum.Data().([]float32))

		var gotIt bool
		for i := 0; i < n; i++ {
			if topClasses[i] == sample.Class {
				gotIt = true
				break
			}
		}

		out <- gotIt
	}
}

func printResults(resChan <-chan bool) {
	var right, total int
	for b := range resChan {
		total++
		if b {
			right++
		}
		fmt.Printf("\rGot %d/%d (%.02f%%)    ", right, total,
			100*float64(right)/float64(total))
	}
}

type valIndexSorter struct {
	Values  []float32
	Indices []int
}

func sortByIndex(vals []float32) []int {
	sorter := &valIndexSorter{Values: vals, Indices: make([]int, len(vals))}
	for i := range vals {
		sorter.Indices[i] = i
	}
	sort.Sort(sorter)
	return sorter.Indices
}

func (s *valIndexSorter) Less(i, j int) bool {
	return s.Values[i] > s.Values[j]
}

func (s *valIndexSorter) Swap(i, j int) {
	s.Values[i], s.Values[j] = s.Values[j], s.Values[i]
	s.Indices[i], s.Indices[j] = s.Indices[j], s.Indices[i]
}

func (s *valIndexSorter) Len() int {
	return len(s.Values)
}
