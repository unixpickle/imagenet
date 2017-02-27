package imagenet

import (
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math/rand"
	"os"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/resize"
)

const (
	InputImageSize   = 224
	MinAugmentedSize = 256
	MaxAugmentedSize = 480
)

// TrainingImage loads the image at the given path and
// transforms it into tensor data.
// It performs various manipulations to the image for the
// purpose of data augmentation.
func TrainingImage(path string) (anyvec.Vector, error) {
	orig, err := readImage(path)
	if err != nil {
		return nil, essentials.AddCtx("load training image:", err)
	}
	img := augmentedImage(orig)
	colorAugment(img)
	return anyvec32.MakeVectorData(img), nil
}

// TestingImages produces tensors for different crops of
// the image.
func TestingImages(path string) ([]anyvec.Vector, error) {
	img, err := readImage(path)
	if err != nil {
		return nil, essentials.AddCtx("load testing images:", err)
	}
	smallerDim := img.Bounds().Dx()
	if img.Bounds().Dy() < smallerDim {
		smallerDim = img.Bounds().Dy()
	}
	var images [][]float32
	for _, size := range []float64{224, 256, 384, 480, 640} {
		scale := size / float64(smallerDim)
		newImage := resize.Resize(uint(float64(img.Bounds().Dx())*scale+0.5),
			uint(float64(img.Bounds().Dy())*scale+0.5), img, resize.Bilinear)
		images = append(images,
			// Top left
			crop(newImage, 0, 0, false),
			// Center
			crop(newImage, (newImage.Bounds().Dx()-InputImageSize)/2,
				(newImage.Bounds().Dy()-InputImageSize)/2, false),
			// Bottom right
			crop(newImage, newImage.Bounds().Dx()-InputImageSize,
				newImage.Bounds().Dy()-InputImageSize, false),
			// Bottom left
			crop(newImage, 0, newImage.Bounds().Dy()-InputImageSize, false),
			// Top right
			crop(newImage, newImage.Bounds().Dx()-InputImageSize, 0, false),
		)
	}
	var res []anyvec.Vector
	for _, x := range images {
		res = append(res, anyvec32.MakeVectorData(x))
	}
	return res, nil
}

func readImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	return img, nil
}

func augmentedImage(img image.Image) []float32 {
	smallerDim := img.Bounds().Dx()
	if img.Bounds().Dy() < smallerDim {
		smallerDim = img.Bounds().Dy()
	}

	// Scale augmentation
	newSize := rand.Intn(MaxAugmentedSize-MinAugmentedSize+1) + MinAugmentedSize
	scale := float64(newSize) / float64(smallerDim)
	newImage := resize.Resize(uint(float64(img.Bounds().Dx())*scale+0.5),
		uint(float64(img.Bounds().Dy())*scale+0.5), img, resize.Bilinear)

	cropX := rand.Intn(newImage.Bounds().Dx() - InputImageSize + 1)
	cropY := rand.Intn(newImage.Bounds().Dy() - InputImageSize + 1)
	return crop(newImage, cropX, cropY, rand.Intn(2) == 1)
}

func crop(img image.Image, cropX, cropY int, mirror bool) []float32 {
	resSlice := make([]float32, 0, InputImageSize*InputImageSize*3)
	for y := 0; y < InputImageSize; y++ {
		for x := 0; x < InputImageSize; x++ {
			sourceX := x
			if mirror {
				sourceX = InputImageSize - (x + 1)
			}
			c := img.At(cropX+sourceX+img.Bounds().Min.X, cropY+y+img.Bounds().Min.Y)
			r, g, b, _ := c.RGBA()
			resSlice = append(resSlice, float32(r)/0xffff, float32(g)/0xffff,
				float32(b)/0xffff)
		}
	}
	return resSlice
}

func colorAugment(t []float32) {
	amount := float32(rand.NormFloat64())

	// Vector from https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4.
	vec := []float32{0.0148366, 0.01253134, 0.01040762}

	for i := 0; i < len(t); i++ {
		t[i] += vec[i%3] * amount
	}
}
