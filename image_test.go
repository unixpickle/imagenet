package imagenet

import (
	"image"
	"image/jpeg"
	"io/ioutil"
	"os"
	"testing"
)

func BenchmarkTrainingImage(b *testing.B) {
	img := image.NewRGBA(image.Rect(0, 0, InputImageSize, InputImageSize))
	w, err := ioutil.TempFile("", "imagenet_test")
	if err != nil {
		b.Fatal(err)
	}
	defer os.Remove(w.Name())
	if err := jpeg.Encode(w, img, nil); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		TrainingImage(w.Name())
	}
}
