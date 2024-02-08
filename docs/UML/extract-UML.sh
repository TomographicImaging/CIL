#!/usr/bin/env bash

# Create a directory where the output files will be generated
if [ ! -d output ]
then
    mkdir output
fi

# Process every package
for dir in ../../Wrappers/Python/cil/*/ 
do
    echo ${dir}
    dir=${dir%*/}
    pyreverse -A -k -o dot ${dir} > /dev/null
    pyreverse -A -k -o png ${dir} > /dev/null
    out=$(basename $dir)
    mv classes.png output/${out}-classes.png
    mv classes.dot output/${out}-classes.dot
    mv packages.png output/${out}-packages.png
    mv packages.dot output/${out}-packages.dot
done

# Create the files for CIL as a whole
pyreverse -A -k -o dot ../../Wrappers/Python/cil > /dev/null
pyreverse -A -k -o plain ../../Wrappers/Python/cil > /dev/null
pyreverse -A -k -o png ../../Wrappers/Python/cil > /dev/null
mv packages.png output/overall-CIL-packages.png
mv packages.dot output/overall-CIL-packages.dot
mv packages.plain output/overall-CIL-packages.plain
mv classes.png output/overall-CIL-classes.png
mv classes.dot output/overall-CIL-classes.dot
rm -f classes.plain