#!/bin/bash

for value in 0.025 0.05 0.0725 0.1 0.125 0.15 0.1725 0.2 0.225 0.25 0.275 0.3 0.325 0.35 0.375 0.4 0.425 0.475 0.5 0.525 0.55 0.575 0.60
do
    echo $'tolerance,'$value >> ./result.txt
    face_recognition --tolerance $value ../data/faces/ ../data/train/ >> ./result.txt 
done

echo All done