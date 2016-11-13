package imagenet

import (
	"crypto/md5"
	"errors"
	"io/ioutil"
	"path/filepath"
	"sort"
	"strings"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// A Sample stores the metadata for a training image.
type Sample struct {
	ClassCount int
	Class      int
	Path       string
}

// A SampleSet is a lazy collection of image samples.
type SampleSet []Sample

// NewSampleSet creates a samlpe set based on the
// directory/file structure of the given root sample
// directory.
func NewSampleSet(dir string) (SampleSet, error) {
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

	var res SampleSet
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
func (s SampleSet) ClassCount() int {
	return s[0].ClassCount
}

func (s SampleSet) Len() int {
	return len(s)
}

func (s SampleSet) Copy() sgd.SampleSet {
	res := make(SampleSet, len(s))
	copy(res, s)
	return res
}

func (s SampleSet) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s SampleSet) GetSample(idx int) interface{} {
	outVec := make(linalg.Vector, s[idx].ClassCount)
	outVec[s[idx].Class] = 1
	sample := neuralnet.VectorSample{
		Input:  TrainingImage(s[idx].Path),
		Output: outVec,
	}
	return sample
}

func (s SampleSet) Subset(start, end int) sgd.SampleSet {
	return s[start:end]
}

// Hash returns the hash of the given sample's base
// filename (e.g. "apple1.png").
func (s SampleSet) Hash(idx int) []byte {
	name := filepath.Base(s[idx].Path)
	hash := md5.Sum([]byte(name))
	return hash[:]
}
