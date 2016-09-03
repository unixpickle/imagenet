package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
)

const (
	WnidFileArg = 1
	ImgCountArg = 2
	DirOutArg   = 3
)

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "wnids img_count dir_out")
		fmt.Fprintln(os.Stderr, "  wnids      file with space-separated wnids")
		fmt.Fprintln(os.Stderr, "  img_count  number of images per wnid")
		fmt.Fprintln(os.Stderr, "  dir_out    output directory")
		os.Exit(1)
	}

	imgCount, err := strconv.Atoi(os.Args[ImgCountArg])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Invalid image count:", os.Args[ImgCountArg])
		os.Exit(1)
	}

	wnidData, err := ioutil.ReadFile(os.Args[WnidFileArg])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to read wnids:", err)
		os.Exit(1)
	}
	wnids := strings.Fields(string(wnidData))

	outDir := os.Args[DirOutArg]
	if statRes, err := os.Stat(outDir); err != nil && os.IsNotExist(err) {
		if err := os.Mkdir(outDir, 0755); err != nil {
			fmt.Fprintln(os.Stderr, "Failed to create output:", err)
			os.Exit(1)
		}
	} else if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	} else if !statRes.IsDir() {
		fmt.Fprintln(os.Stderr, "Not a directory:", outDir)
		os.Exit(1)
	}

	Fetch(wnids, imgCount, outDir)
}
