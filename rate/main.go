package main

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"sort"
	"strconv"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"

	_ "github.com/unixpickle/batchnorm"
)

func main() {
	if len(os.Args) != 3 && len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "model_file image_dir [n]")
		os.Exit(1)
	}

	n := 1
	if len(os.Args) == 4 {
		count, err := strconv.Atoi(os.Args[3])
		if err != nil {
			fmt.Fprintln(os.Stderr, "Invalid top-n value:", os.Args[3])
			os.Exit(1)
		}
		n = count
	}

	model, err := readModel()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read model:", err)
		os.Exit(1)
	}

	samples, err := imagenet.NewSampleSet(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load samples:", err)
		os.Exit(1)
	}

	rand.Seed(time.Now().UnixNano())
	sgd.ShuffleSampleSet(samples)

	sampleChan := make(chan neuralnet.VectorSample, 1)
	go func() {
		for i := 0; i < samples.Len(); i++ {
			sampleChan <- samples.GetSample(i).(neuralnet.VectorSample)
		}
		close(sampleChan)
	}()

	outChan := make(chan bool)

	go func() {
		rateSamples(n, model, sampleChan, outChan)
		close(outChan)
	}()

	printResults(outChan)
}

func rateSamples(n int, net neuralnet.Network, samples <-chan neuralnet.VectorSample,
	out chan<- bool) {
	for sample := range samples {
		res := net.Apply(&autofunc.Variable{Vector: sample.Input}).Output()

		topClasses := sortByIndex(res)
		var gotIt bool
		for i := 0; i < n; i++ {
			if sample.Output[topClasses[i]] != 0 {
				gotIt = true
				break
			}
		}

		out <- gotIt
	}
}

func readModel() (neuralnet.Network, error) {
	modelData, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		return nil, err
	}
	return neuralnet.DeserializeNetwork(modelData)
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
	Values  []float64
	Indices []int
}

func sortByIndex(vals []float64) []int {
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
