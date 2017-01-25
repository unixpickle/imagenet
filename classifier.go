package imagenet

import (
	"encoding/json"
	"errors"

	"github.com/unixpickle/anynet"
	"github.com/unixpickle/serializer"
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
