

#! /bin/bash


name=cil
python=3.8
numpy=1.20

while getopts h:n:p:e:v: option
 do
 case "${option}"
 in
  n) numpy=${OPTARG};;
  p) python=${OPTARG};;
  e) name=${OPTARG};;
  v) ver=${OPTARG};;
  h)
   echo "Usage: $0 [-n numpy_version] [-p python version] [-e environment name] [-v optional cil version]"
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

if [[ -n $ver ]]
then
    echo CIL version $ver
    conda_cmd="conda create --name ${name} cil=${ver}"
else
    conda_cmd="conda create --name ${name}"

fi

set -x 

${conda_cmd} -c conda-forge -c intel  -c ccpi/label/dev -c ccpi -c astra-toolbox -c astra-toolbox/label/dev \
        python=$python numpy=$numpy \
        cil-data tigre=2.2 ccpi-regulariser=21.0.0 tomophantom=2.0.0  astra-toolbox'>=1.9.9.dev5,<2.1' \
        cvxpy python-wget scikit-image packaging \
        cmake'>=3.16' setuptools  \
        ipp-include ipp-devel ipp \
        ipywidgets scipy matplotlib \
        h5py pillow libgcc-ng dxchange olefile pywavelets numba tqdm \
        --override-channels
