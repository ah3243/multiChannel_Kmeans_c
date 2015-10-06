#!/bin/bash
set -e

cd build2/
cmake ..
make
echo made and built
~/Desktop/TEST_IMAGES/CapturedImgs/./ranImgs.sh

#Attempts=(5 10 15 20)
Scale=(9)

#inputType="attempts"
#inputType="texDict"
inputType="scale"

for i in ${Scale[@]}; do

	#Note 1.inputValue  2.Input Type
	./multiDimen ${i} $inputType
	echo Test Type is: $inputType
	echo iteration ${i}
done
echo Done
