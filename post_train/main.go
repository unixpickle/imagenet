package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/signal"
	"runtime"
	"sync"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/batchnorm"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	ImageDirArg = 1
	OutNetArg   = 2
)

type ActivationInfo struct {
	Value  linalg.Vector
	Square linalg.Vector
	Layer  *batchnorm.Layer
}

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "image_dir net_file")
		os.Exit(1)
	}

	log.Println("Loading samples...")
	samples, err := imagenet.NewSampleSet(os.Args[ImageDirArg])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read sample listing:", err)
		os.Exit(1)
	}

	log.Println("Loading network...")
	netData, err := ioutil.ReadFile(os.Args[OutNetArg])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read network:", err)
		os.Exit(1)
	}
	net, err := neuralnet.DeserializeNetwork(netData)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to deserialize network:", err)
		os.Exit(1)
	}

	log.Println("Computing means...")

	ins := make(chan linalg.Vector, 1)
	go func() {
		defer close(ins)

		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt)
		for i := 0; i < samples.Len(); i++ {
			select {
			case <-c:
				return
			default:
			}
			ins <- samples.GetSample(i).(neuralnet.VectorSample).Input
		}
	}()

	var wg sync.WaitGroup
	outs := make(chan *ActivationInfo, 1)

	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go computeMeans(net, ins, outs, &wg)
	}

	go func() {
		wg.Wait()
		close(outs)
	}()

	var processedCount int
	counts := map[*batchnorm.Layer]int{}
	for a := range outs {
		processedCount++
		counts[a.Layer] += addMoments(a)
		fmt.Printf("\rProcessed %d activations.", processedCount)
	}
	fmt.Println()

	for layer, count := range counts {
		layer.DoneTraining = true
		layer.FinalMean.Scale(1 / float64(count))
		layer.FinalVariance.Scale(1 / float64(count))
		layer.FinalVariance.Add(square(layer.FinalMean).Scale(-1))
	}

	log.Println("Saving network...")

	data, err := net.Serialize()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to serialize:", err)
		os.Exit(1)
	}

	if err := ioutil.WriteFile(os.Args[OutNetArg], data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save network:", err)
		os.Exit(1)
	}
}

func computeMeans(net neuralnet.Network, ins <-chan linalg.Vector,
	outs chan<- *ActivationInfo, wg *sync.WaitGroup) {
	for input := range ins {
		for _, layer := range net {
			if l, ok := layer.(*batchnorm.Layer); ok {
				outs <- &ActivationInfo{
					Value:  input.Copy(),
					Square: square(input),
					Layer:  l,
				}
			}
			input = layer.Apply(&autofunc.Variable{Vector: input}).Output()
		}
	}
	wg.Done()
}

func square(in linalg.Vector) linalg.Vector {
	res := make(linalg.Vector, len(in))
	for i, x := range in {
		res[i] = x * x
	}
	return res
}

func addMoments(a *ActivationInfo) int {
	var count int
	for i := 0; i < len(a.Value); i += a.Layer.InputCount {
		val := a.Value[i : i+a.Layer.InputCount]
		square := a.Square[i : i+a.Layer.InputCount]
		if a.Layer.FinalMean == nil {
			a.Layer.FinalMean = val
			a.Layer.FinalVariance = square
		} else {
			a.Layer.FinalMean.Add(val)
			a.Layer.FinalVariance.Add(square)
		}
		count++
	}
	return count
}
