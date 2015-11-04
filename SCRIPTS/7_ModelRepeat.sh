#!/bin/bash
set -e

###################################################################
###                 TRAINING REPEAT VARIATION                   ###
###################################################################

## cd to build dir and make
cd ../buildTRAINING/
cmake ..
make
echo made and built

## VARIABLE TYPES
inputType="modelRepeats"

## SCALE DEPENDANT VALS
	# Scale=7
  # CropSize=113

  Scale=8
  CropSize=71

firstGo=1 # Prevents results labels being printed after the first iteration
counter=0

Training=(1 2 4 6 8 10) # Number of Training Model Repeats

for i in ${Training[@]}; do
	# If not first iteration then set flag as 0
	if [ $counter -gt 0 ]
	then
	  echo not first iteration
	  firstGo=0
	fi

	echo starting program

	#Note 1.inputValue  2.Input Type 3.firstLineFlag
	echo Test Type is: $inputType ModelRepeats: ${i} First go?: $firstGo Scale: $Scale CropSize: $CropSize
	echo iteration $counter
	echo
	./multiDimen ${i} $inputType $firstGo $Scale $CropSize
	let counter=counter+1
done
