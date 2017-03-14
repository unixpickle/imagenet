#!/bin/bash

fetch() {
  PAGE_URL="$1"
  OUT_TXT="$2".txt
  OUT_JSON="$2".json

  echo 'Fetching JSON data...'

  echo "{" >$OUT_JSON
  curl -s "$PAGE_URL" | grep 'synset?wnid=' |
    sed -E $'s/<\\/a>/\\\n/g' | grep 'wnid=' |
    sed -E 's/^.*wnid=([a-z0-9]*)">(.*)$/  "\1": "\2",/g' |
    tr $'\n' '~' | sed -E 's/,~$//g' | tr '~' $'\n' >>$OUT_JSON
  echo "}" >>$OUT_JSON

  echo 'Building text list...'
  cat $OUT_JSON | grep '"' | sed -E 's/^  "(.*)": .*$/\1/g' | sort >$OUT_TXT
}

fetch 'http://image-net.org/challenges/LSVRC/2014/browse-synsets' 'ilsvrc_2014'
fetch 'http://image-net.org/challenges/LSVRC/2010/browse-synsets' 'ilsvrc_2010'
