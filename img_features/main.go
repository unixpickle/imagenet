// Command img_features runs an image through a network
// and computes the average activation at some layer
// for different croppings of the image.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/serializer"
)

func main() {
	var truncLayers int
	var centerOnly bool
	var fiveCrop bool

	flag.IntVar(&truncLayers, "layers", 2, "number of output layers to ignore")
	flag.BoolVar(&centerOnly, "center", false, "only use one (centered) crop")
	flag.BoolVar(&fiveCrop, "fivecrop", false, "only use five crops")
	flag.Parse()

	if len(flag.Args()) < 2 {
		essentials.Die("Usage: img_features [flags] net_file images...")
	}
	netPath := flag.Args()[0]
	imagePaths := flag.Args()[1:]

	var classifier *imagenet.Classifier
	if err := serializer.LoadAny(netPath, &classifier); err != nil {
		essentials.Die(err)
	}
	if truncLayers > len(classifier.Net) {
		essentials.Die("cannot remove", truncLayers, "layers")
	}
	net := classifier.Net[:len(classifier.Net)-truncLayers]

	for _, path := range imagePaths {
		var croppings []anyvec.Vector
		if centerOnly {
			img, err := imagenet.TestingCenterImage(path)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				continue
			}
			croppings = []anyvec.Vector{img}
		} else {
			var err error
			croppings, err = imagenet.TestingImages(path)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				continue
			}
			if fiveCrop {
				croppings = croppings[:5]
			}
		}

		var sum anyvec.Vector
		for _, cropping := range croppings {
			out := net.Apply(anydiff.NewConst(cropping), 1)
			if sum == nil {
				sum = out.Output().Copy()
			} else {
				sum.Add(out.Output())
			}
		}
		sum.Scale(1 / float32(len(croppings)))

		for i, x := range sum.Data().([]float32) {
			if i > 0 {
				fmt.Print(" ")
			}
			fmt.Printf("%f", x)
		}
		fmt.Println()
	}
}
