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
	resSlice := imageToTensor(img)
	colorAugment(resSlice)
	return anyvec32.MakeVectorData(resSlice), nil
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
	var images []image.Image
	for _, size := range []float64{224, 256, 384, 480, 640} {
		scale := size / float64(smallerDim)
		newImage := resize.Resize(uint(float64(img.Bounds().Dx())*scale+0.5),
			uint(float64(img.Bounds().Dy())*scale+0.5), img, resize.Bilinear)
		images = append(images,
			// Top left
			crop(newImage, 0, 0),
			// Center
			crop(newImage, (newImage.Bounds().Dx()-InputImageSize)/2,
				(newImage.Bounds().Dy()-InputImageSize)/2),
			// Bottom right
			crop(newImage, newImage.Bounds().Dx()-InputImageSize,
				newImage.Bounds().Dy()-InputImageSize),
			// Bottom left
			crop(newImage, 0, newImage.Bounds().Dy()-InputImageSize),
			// Top right
			crop(newImage, newImage.Bounds().Dx()-InputImageSize, 0),
		)
	}
	var res []anyvec.Vector
	for _, x := range images {
		res = append(res, anyvec32.MakeVectorData(imageToTensor(x)))
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

func imageToTensor(img image.Image) []float32 {
	resSlice := make([]float32, InputImageSize*InputImageSize*3)
	for y := 0; y < InputImageSize; y++ {
		for x := 0; x < InputImageSize; x++ {
			pixel := img.At(x+img.Bounds().Min.X, y+img.Bounds().Min.Y)
			idx := (y*InputImageSize + x) * 3
			r, g, b, _ := pixel.RGBA()
			resSlice[idx] = float32(r) / 0xffff
			resSlice[idx+1] = float32(g) / 0xffff
			resSlice[idx+2] = float32(b) / 0xffff
		}
	}
	return resSlice
}

func augmentedImage(img image.Image) image.Image {
	smallerDim := img.Bounds().Dx()
	if img.Bounds().Dy() < smallerDim {
		smallerDim = img.Bounds().Dy()
	}

	// Scale augmentation
	newSize := rand.Intn(MaxAugmentedSize-MinAugmentedSize+1) + MinAugmentedSize
	scale := float64(newSize) / float64(smallerDim)
	newImage := resize.Resize(uint(float64(img.Bounds().Dx())*scale+0.5),
		uint(float64(img.Bounds().Dy())*scale+0.5), img, resize.Bilinear)

	if rand.Intn(2) == 0 {
		newImage = mirrorImage(newImage)
	}

	cropX := rand.Intn(newImage.Bounds().Dx() - InputImageSize + 1)
	cropY := rand.Intn(newImage.Bounds().Dy() - InputImageSize + 1)
	return crop(newImage, cropX, cropY)
}

func crop(img image.Image, cropX, cropY int) image.Image {
	res := image.NewRGBA(image.Rect(0, 0, InputImageSize, InputImageSize))
	for y := 0; y < InputImageSize; y++ {
		for x := 0; x < InputImageSize; x++ {
			c := img.At(cropX+x+img.Bounds().Min.X, cropY+y+img.Bounds().Min.Y)
			res.Set(x, y, c)
		}
	}
	return res
}

func mirrorImage(img image.Image) image.Image {
	res := image.NewRGBA(image.Rect(0, 0, img.Bounds().Dx(), img.Bounds().Dy()))
	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			c := img.At(x+img.Bounds().Min.X, y+img.Bounds().Min.Y)
			res.Set(img.Bounds().Dx()-(x+1), y, c)
		}
	}
	return res
}

func colorAugment(t []float32) {
	amount := float32(rand.NormFloat64())

	// Vector from https://groups.google.com/forum/#!topic/lasagne-users/meCDNeA9Ud4.
	vec := []float32{0.0148366, 0.01253134, 0.01040762}

	for i := 0; i < len(t); i++ {
		t[i] += vec[i%3] * amount
	}
}
