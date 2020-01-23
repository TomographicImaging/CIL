if [ -z "$CIL_VERSION" ]; then
    echo "Need to set CIL_VERSION"
    exit 1
fi
# mkdir ${SRC_DIR}/ccpi
mkdir -p ${SRC_DIR}/ccpi/Wrappers/Python
cp -r "${RECIPE_DIR}/../Wrappers/Python/test" ${SRC_DIR}/ccpi/Wrappers/Python

# cd ${SRC_DIR}/ccpi/Wrappers/Python
# $PYTHON setup.py install


mkdir ${SRC_DIR}/build_framework
#cp -r "${RECIPE_DIR}/../" ${SRC_DIR}/build_framework

cd ${SRC_DIR}/build_framework

if [ `python -c "from __future__ import print_function; import platform; print (platform.system())"`  == "Darwin" ] ;
then 
  echo "Darwin"; 
  cmake ${RECIPE_DIR}/../ -DCONDA_BUILD=ON \
                        -DCMAKE_BUILD_TYPE="Release"\
                        -DLIBRARY_LIB=$CONDA_PREFIX/lib \
                        -DLIBRARY_INC=$CONDA_PREFIX \
                        -DCMAKE_INSTALL_PREFIX=$PREFIX\
                        -DOPENMP_INCLUDES=${CONDA_PREFIX}/include \
                        -DOPENMP_LIBRARIES=${CONDA_PREFIX}/lib
else 
  echo "something else"; 

  cmake ${RECIPE_DIR}/../ -DCONDA_BUILD=ON \
                        -DCMAKE_BUILD_TYPE="Release"\
                        -DLIBRARY_LIB=$CONDA_PREFIX/lib \
                        -DLIBRARY_INC=$CONDA_PREFIX \
                        -DCMAKE_INSTALL_PREFIX=$PREFIX

fi
make install VERBOSE=1
# $PYTHON setup.py install
