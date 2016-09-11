package main

import (
	"bytes"
	"errors"
	"fmt"
	"image/jpeg"
	"image/png"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

const FetchRoutines = 30
const ImageNetAPI = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="

func Fetch(wnids []string, imgCount int, outDir string) {
	wnidChan := make(chan string, len(wnids))
	for _, w := range wnids {
		wnidChan <- w
	}
	close(wnidChan)

	var wg sync.WaitGroup
	for i := 0; i < FetchRoutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for wnid := range wnidChan {
				fetchWNID(wnid, imgCount, outDir)
			}
		}()
	}
	wg.Wait()
}

func fetchWNID(wnid string, maxCount int, outDir string) {
	log.Println("Fetching images for wnid:", wnid)
	defer log.Println("Done with", wnid)

	subPath := filepath.Join(outDir, wnid)

	statRes, err := os.Stat(subPath)
	if err != nil && os.IsNotExist(err) {
		if err := os.Mkdir(subPath, 0755); err != nil {
			log.Printf("Error making directory %s: %s", subPath, err)
			return
		}
	} else if err != nil {
		log.Printf("Failed to stat %s: %s", subPath, err)
		return
	} else if !statRes.IsDir() {
		log.Println("Path is not a directory:", subPath)
		return
	}
	dirListing, err := ioutil.ReadDir(subPath)
	if err != nil {
		log.Printf("Failed to readdir %s: %s", subPath, err)
		return
	}

	var imageCount int
	for _, item := range dirListing {
		if strings.HasPrefix(item.Name(), ".") {
			continue
		}
		imageCount++
	}

	if imageCount >= maxCount {
		return
	}

	urls, err := urlsForWNID(wnid)
	if err != nil {
		log.Printf("Failed to lookup wnid %s: %s", wnid, err)
		return
	}

	for _, i := range rand.Perm(len(urls)) {
		if imageCount >= maxCount {
			break
		}
		data, ext, err := fetchImage(urls[i])
		if err != nil {
			continue
		}
		imgPath := filepath.Join(subPath, fmt.Sprintf("%d.%s", rand.Int(), ext))
		if err := ioutil.WriteFile(imgPath, data, 0755); err != nil {
			log.Printf("Failed to write %s: %s", imgPath, err)
			return
		}
		imageCount++
	}
}

func urlsForWNID(wnid string) ([]string, error) {
	res, err := http.Get(ImageNetAPI + wnid)
	if err != nil {
		return nil, err
	}
	contents, err := ioutil.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		return nil, err
	}
	return strings.Fields(string(contents)), nil
}

func fetchImage(fileURL string) (data []byte, extension string, err error) {
	res, err := http.Get(fileURL)
	if err != nil {
		return nil, "", err
	}
	contents, err := ioutil.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		return nil, "", err
	}

	if _, err := jpeg.Decode(bytes.NewBuffer(contents)); err == nil {
		return contents, "jpg", nil
	}
	if _, err := png.Decode(bytes.NewBuffer(contents)); err == nil {
		return contents, "png", nil
	}
	return nil, "", errors.New("unsupported image format")
}
