#!/bin/bash

# use absolute path
for filename in /mnt/c/Users/Deng/Desktop/bw_out/*.jpg; do
    [ -e "$filename" ] || continue
    ./Locator "$filename"
done
