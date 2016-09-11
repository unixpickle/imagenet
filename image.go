package imagenet

import (
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/resize"
	"github.com/unixpickle/weakai/neuralnet"
)

const InputImageSize = 224

// TrainingImage loads the image at the given path and
// transforms it into tensor data.
func TrainingImage(path string) linalg.Vector {
	f, err := os.Open(path)
	if err != nil {
		panic("could not read training image: " + path)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		panic("could not decode training image: " + path)
	}
	smallerDim := img.Bounds().Dx()
	if img.Bounds().Dy() < smallerDim {
		smallerDim = img.Bounds().Dy()
	}
	scale := InputImageSize / float64(smallerDim)
	newImage := resize.Resize(uint(float64(img.Bounds().Dx())*scale+0.5),
		uint(float64(img.Bounds().Dy())*scale+0.5), img, resize.Bilinear)

	cropX := (newImage.Bounds().Dx() - InputImageSize) / 2
	cropY := (newImage.Bounds().Dy() - InputImageSize) / 2

	res := neuralnet.NewTensor3(InputImageSize, InputImageSize, 3)
	for x := 0; x < InputImageSize; x++ {
		for y := 0; y < InputImageSize; y++ {
			pixel := newImage.At(x+newImage.Bounds().Min.X+cropX,
				y+newImage.Bounds().Min.Y+cropY)
			r, g, b, _ := pixel.RGBA()
			res.Set(x, y, 0, float64(r)/0xffff)
			res.Set(x, y, 1, float64(g)/0xffff)
			res.Set(x, y, 2, float64(b)/0xffff)
		}
	}

	return res.Data
}
