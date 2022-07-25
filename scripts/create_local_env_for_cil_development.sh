#! /bin/bash

while getopts hn:p:e: option
 do
 case "${option}"
 in
  n) numpy=${OPTARG};;
  p) python=${OPTARG};;
  e) name=${OPTARG};;
  h)
   echo "Usage: $0 [-n numpy_version] [-p python version] [-e environment name]"
   exit 
   ;;
  *)
   echo "Wrong option passed. Use the -h option to get some help." >&2
   exit 1
  ;;
 esac
done

echo Numpy $numpy
echo Python $python
echo Environment name $name

set -x

conda create --name $name cmake python=$python numpy=$numpy scipy matplotlib \
  h5py pillow libgcc-ng dxchange olefile pywavelets python-wget scikit-image \
  packaging  numba ipp ipp-devel ipp-include tqdm -c conda-forge -c intel  -c defaults --override-channels
