package imagenet

import (
	"image"
	"image/color"
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
		return nil, essentials.AddCtx("read image "+path, err)
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
		return nil, essentials.AddCtx("read image "+path, err)
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

// TestingCenterImage crops the center of the image and
// returns it as a tensor.
func TestingCenterImage(path string) (anyvec.Vector, error) {
	img, err := readImage(path)
	if err != nil {
		return nil, essentials.AddCtx("read image "+path, err)
	}
	return ImageToTensor(img), nil
}

// ImageToTensor converts an image to a tensor.
//
// The image is scaled and cropped (in the center) so that
// the output tensor has the right dimensions.
// If the image is already InputImageSize on both sides,
// then it is not cropped or scaled.
func ImageToTensor(img image.Image) anyvec.Vector {
	smallerDim := img.Bounds().Dx()
	if img.Bounds().Dy() < smallerDim {
		smallerDim = img.Bounds().Dy()
	}
	scale := InputImageSize / float64(smallerDim)
	newImage := resize.Resize(uint(float64(img.Bounds().Dx())*scale+0.5),
		uint(float64(img.Bounds().Dy())*scale+0.5), img, resize.Bilinear)
	slice := crop(newImage, (newImage.Bounds().Dx()-InputImageSize)/2,
		(newImage.Bounds().Dy()-InputImageSize)/2, false)
	return anyvec32.MakeVectorData(slice)
}

// TensorToImage converts a tensor to an image.
func TensorToImage(tensor anyvec.Vector) image.Image {
	data := tensor.Data().([]float32)
	res := image.NewRGBA(image.Rect(0, 0, InputImageSize, InputImageSize))
	for y := 0; y < res.Bounds().Dy(); y++ {
		for x := 0; x < res.Bounds().Dx(); x++ {
			idx := 3 * (x + y*res.Bounds().Dx())
			red := uint8(data[idx]*0xff + 0.5)
			green := uint8(data[idx+1]*0xff + 0.5)
			blue := uint8(data[idx+2]*0xff + 0.5)
			res.SetRGBA(x, y, color.RGBA{
				R: red,
				G: green,
				B: blue,
				A: 0xff,
			})
		}
	}
	return res
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
