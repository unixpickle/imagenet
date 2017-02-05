package imagenet

import (
	"crypto/md5"
	"errors"
	"io/ioutil"
	"path/filepath"
	"sort"
	"strings"

	"github.com/unixpickle/anynet/anyff"
	"github.com/unixpickle/anynet/anysgd"
)

// A Sample stores the metadata for a training image.
type Sample struct {
	ClassCount int
	Class      int
	Path       string
}

// A SampleList is a lazy collection of image samples.
type SampleList []Sample

// NewSampleList creates a samlpe set based on the
// directory/file structure of the given root sample
// directory.
func NewSampleList(dir string) (SampleList, error) {
	imageDirs, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var dirNames []string
	for _, item := range imageDirs {
		if !item.IsDir() {
			continue
		}
		dirNames = append(dirNames, filepath.Join(dir, item.Name()))
	}
	sort.Strings(dirNames)

	var res SampleList
	var class int
	for _, subDir := range dirNames {
		listing, err := ioutil.ReadDir(subDir)
		if err != nil {
			return nil, err
		}
		for _, fileItem := range listing {
			if strings.HasPrefix(fileItem.Name(), ".") {
				continue
			}
			res = append(res, Sample{
				ClassCount: len(dirNames),
				Class:      class,
				Path:       filepath.Join(subDir, fileItem.Name()),
			})
		}
		class++
	}
	if len(res) == 0 {
		return nil, errors.New("no training images found")
	}
	return res, nil
}

// ClassCount returns the number of classes in the set.
func (s SampleList) ClassCount() int {
	return s[0].ClassCount
}

func (s SampleList) Len() int {
	return len(s)
}

func (s SampleList) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s SampleList) Slice(start, end int) anysgd.SampleList {
	return append(SampleList{}, s[start:end]...)
}

func (s SampleList) GetSample(idx int) *anyff.Sample {
	return s.getSample(false, idx)
}

func (s SampleList) GetCenteredSample(idx int) *anyff.Sample {
	return s.getSample(true, idx)
}

// Hash returns the hash of the given sample's base
// filename (e.g. "apple1.png").
func (s SampleList) Hash(idx int) []byte {
	name := filepath.Base(s[idx].Path)
	hash := md5.Sum([]byte(name))
	return hash[:]
}

func (s SampleList) getSample(center bool, idx int) *anyff.Sample {
	outVec := make([]float64, s[idx].ClassCount)
	outVec[s[idx].Class] = 1
	in := TrainingImage(center, s[idx].Path)
	sample := &anyff.Sample{
		Input:  in,
		Output: in.Creator().MakeVectorData(in.Creator().MakeNumericList(outVec)),
	}
	return sample
}
