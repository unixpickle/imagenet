package imagenet

import (
	"encoding/json"
	"errors"
	"sort"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/serializer"

	_ "github.com/unixpickle/anynet/anyconv"
)

func init() {
	var c Classifier
	serializer.RegisterTypedDeserializer(c.SerializerType(), DeserializeClassifier)
}

// A Classifier is a trained feed-forward classifier.
type Classifier struct {
	InWidth  int
	InHeight int

	// Net takes input tensors and produces the logarithms of
	// classification probabilities.
	Net anynet.Net

	// For each network output, Classes contains a textual
	// identifier for it.
	// This may be a word, a WordNet ID, or something else.
	Classes []string
}

// DeserializeClassifier deserializes a Classifier.
func DeserializeClassifier(d []byte) (*Classifier, error) {
	var net anynet.Net
	var metaData serializer.Bytes
	if err := serializer.DeserializeAny(d, &net, &metaData); err != nil {
		return nil, errors.New("deserialize classifier: " + err.Error())
	}
	var cs classifierSaver
	if err := json.Unmarshal(metaData, &cs); err != nil {
		return nil, errors.New("deserialize classifier: " + err.Error())
	}
	return &Classifier{
		InWidth:  cs.InWidth,
		InHeight: cs.InHeight,
		Net:      net,
		Classes:  cs.Classes,
	}, nil
}

// Classify classifies an image and returns a list of
// classes and their corresponding confidences.
// The result is sorted most-to-least probable.
// The input is one or more croppings of the image, such
// as the croppings returned by TestingImages().
func (c *Classifier) Classify(versions []anyvec.Vector) ([]string, []float64) {
	joinedIn := versions[0].Creator().Concat(versions...)
	out := c.Net.Apply(anydiff.NewConst(joinedIn), len(versions)).Output()
	anyvec.Exp(out)
	sum := anyvec.SumRows(out, out.Len()/len(versions))
	sum.Scale(1 / float32(len(versions)))
	probs := sum.Data().([]float32)

	sorter := &probSorter{
		Probs:   probs,
		Classes: append([]string{}, c.Classes...),
	}
	sort.Sort(sorter)

	probs64 := make([]float64, len(sorter.Probs))
	for i, x := range sorter.Probs {
		probs64[i] = float64(x)
	}
	return sorter.Classes, probs64
}

// SerializerType returns the unique ID used to serialize
// a Classifier with the serializer package.
func (c *Classifier) SerializerType() string {
	return "github.com/unixpickle/imagenet.Classifier"
}

// Serialize serializes the Classifier.
func (c *Classifier) Serialize() ([]byte, error) {
	meta := &classifierSaver{
		InWidth:  c.InWidth,
		InHeight: c.InHeight,
		Classes:  c.Classes,
	}
	metaData, err := json.Marshal(meta)
	if err != nil {
		return nil, err
	}
	return serializer.SerializeAny(c.Net, serializer.Bytes(metaData))
}

type classifierSaver struct {
	InWidth  int
	InHeight int
	Classes  []string
}

type probSorter struct {
	Probs   []float32
	Classes []string
}

func (p *probSorter) Len() int {
	return len(p.Probs)
}

func (p *probSorter) Less(i, j int) bool {
	return p.Probs[i] > p.Probs[j]
}

func (p *probSorter) Swap(i, j int) {
	p.Probs[i], p.Probs[j] = p.Probs[j], p.Probs[i]
	p.Classes[i], p.Classes[j] = p.Classes[j], p.Classes[i]
}
