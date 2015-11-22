#!/bin/bash
set -e

############################################################
###                 CROPPING VARIATION                   ###
############################################################


## cd to build dir and make
cd ../build/
cmake ..
make
echo made and built

  # Generate new images if flag input
  if [ "$useImgs" == "Y" ]
  then
	~/Desktop/TEST_IMAGES/CapturedImgs/./ranImgs.sh
	echo images randomised and allocated
  fi

	## VARIABLE TYPES
	inputType="texDict"
	## SCALE VALUES
	Scale=8

  testRepeats=5
  modelRepeats=20

	firstGo=1 # Prevents results labels being printed after the first iteration
	counter=0
  CropSize=70
  texDict=(2 5 8 10 15 20 25 30)

	for i in ${texDict[@]}; do
		# If not first iteration then set flag as 0
		if [ $counter -gt 0 ]
		then
		  echo not first iteration
		  firstGo=0
		fi

		echo starting program

		#Note 1.inputValue  2.Input Type 3.firstLineFlag
		echo Test Type is: $inputType InputParam: ${i} First go?: $firstGo Scale: $Scale testRepeats: $testRepeats
		echo iteration $counter
		echo
		./multiDimen ${i} $inputType $firstGo $Scale $testRepeats $modelRepeats $CropSize
		let counter=counter+1
	done

echo Done
