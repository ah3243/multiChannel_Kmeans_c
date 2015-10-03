#!/bin/bash

echo Running Opencv Attempts Loop

cd build2/
cmake ..
make
echo made and built
~/Desktop/TEST_IMAGES/CapturedImgs/./ranImgs.sh

i=0

while [ $i -lt 1 ] 
do
	./multiDimen
echo iteration $i
	i = i+1
done
echo Done
