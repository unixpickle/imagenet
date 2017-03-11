package main

import (
	"flag"
	_ "image/jpeg"
	"image/png"
	"log"
	"math"
	"os"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/rip"
	"github.com/unixpickle/serializer"
)

func main() {
	var imagePath string
	var netPath string
	var outPath string
	var stepSize float64
	var layer int

	flag.StringVar(&imagePath, "in", "", "input image path")
	flag.StringVar(&netPath, "net", "", "network path")
	flag.StringVar(&outPath, "out", "output.png", "output image path")
	flag.Float64Var(&stepSize, "step", 10, "SGD step size")
	flag.IntVar(&layer, "layer", 21, "layer to maximize")

	flag.Parse()
	if imagePath == "" || netPath == "" {
		essentials.Die("Required flags: -in and -net. See -help.")
	}

	var net *imagenet.Classifier
	if err := serializer.LoadAny(netPath, &net); err != nil {
		essentials.Die("load network:", err)
	}
	if layer >= len(net.Net) {
		essentials.Die("layer out of bounds")
	}
	preLayers := net.Net[:layer]

	tensors, err := imagenet.TestingImages(imagePath)
	if err != nil {
		essentials.Die(err)
	}
	// Take the center crop.
	tensor := tensors[1]

	log.Println("Press ctrl+c to finish...")
	params := anydiff.NewVar(inverseSigmoid(tensor))
	grad := anydiff.NewGrad(params)
	r := rip.NewRIP()
	var iter int
	for !r.Done() {
		input := anydiff.Sigmoid(params)
		output := preLayers.Apply(input, 1)
		zeroVec := anydiff.NewConst(anyvec32.MakeVector(output.Output().Len()))
		cost := anynet.MSE{}.Cost(zeroVec, output, 1)
		grad.Scale(float32(0))
		cost.Propagate(anyvec32.MakeVectorData([]float32{1}), grad)
		grad.Scale(float32(stepSize))
		grad.AddToVars()
		log.Printf("iter %d: cost=%v", iter, anyvec.Sum(cost.Output()))
		iter++
	}

	log.Println("Saving output image...")
	image := imagenet.TensorToImage(anydiff.Sigmoid(params).Output())
	w, err := os.Create(outPath)
	if err != nil {
		essentials.Die(err)
	}
	defer w.Close()
	if err := png.Encode(w, image); err != nil {
		essentials.Die(err)
	}
}

func inverseSigmoid(tensor anyvec.Vector) anyvec.Vector {
	var res []float32
	for _, x := range tensor.Data().([]float32) {
		clipped := math.Min(1-1e-3, math.Max(1e-3, float64(x)))
		res = append(res, float32(-math.Log(1/clipped-1)))
	}
	return anyvec32.MakeVectorData(res)
}
