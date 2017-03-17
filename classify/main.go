package main

import (
	"flag"
	"fmt"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/imagenet"
	"github.com/unixpickle/serializer"
)

func main() {
	var classifierPath string
	var imagePath string
	var numGuesses int
	var printConfidence bool
	var centerOnly bool
	flag.StringVar(&classifierPath, "classifier", "", "path to classifier")
	flag.StringVar(&imagePath, "image", "", "input image")
	flag.IntVar(&numGuesses, "n", 1, "number of guesses")
	flag.BoolVar(&printConfidence, "confidence", false, "print confidence")
	flag.BoolVar(&centerOnly, "center", false, "only use center crop")
	flag.Parse()

	if classifierPath == "" || imagePath == "" {
		essentials.Die("Required flags: -classifier and -image. See -help")
	}

	var classifier *imagenet.Classifier
	if err := serializer.LoadAny(classifierPath, &classifier); err != nil {
		essentials.Die(err)
	}

	var images []anyvec.Vector
	if centerOnly {
		image, err := imagenet.TestingCenterImage(imagePath)
		if err != nil {
			essentials.Die(err)
		}
		images = []anyvec.Vector{image}
	} else {
		var err error
		images, err = imagenet.TestingImages(imagePath)
		if err != nil {
			essentials.Die(err)
		}
	}

	classes, probs := classifier.Classify(images)
	for i := 0; i < len(classes) && i < numGuesses; i++ {
		if printConfidence {
			fmt.Println(classes[i], probs[i])
		} else {
			fmt.Println(classes[i])
		}
	}
}
