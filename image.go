package imagenet

import (
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/resize"
)

const InputImageSize = 224

// TrainingImage loads the image at the given path and
// transforms it into tensor data.
func TrainingImage(path string) anyvec.Vector {
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

	resSlice := make([]float32, InputImageSize*InputImageSize*3)
	for y := 0; y < InputImageSize; y++ {
		for x := 0; x < InputImageSize; x++ {
			pixel := newImage.At(x+newImage.Bounds().Min.X+cropX,
				y+newImage.Bounds().Min.Y+cropY)
			idx := (y*InputImageSize + x) * 3
			r, g, b, _ := pixel.RGBA()
			resSlice[idx] = float32(r) / 0xffff
			resSlice[idx+1] = float32(g) / 0xffff
			resSlice[idx+2] = float32(b) / 0xffff
		}
	}

	return anyvec32.MakeVectorData(resSlice)
}
