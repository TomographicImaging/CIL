mkdir ${SRC_DIR}/ccpi
cp -r "${RECIPE_DIR}/../../../" ${SRC_DIR}/ccpi

#cd ${SRC_DIR}/ccpi/Wrappers/Python
#$PYTHON setup.py install
mkdir ${SRC_DIR}/build/
cd ${SRC_DIR}/build



cmake ../ccpi $RECIPE_DIR/../../../ -DBUILD_PYTHON_WRAPPER=ON -DCONDA_BUILD=ON -DCMAKE_BUILD_TYPE="Release" -DLIBRARY_LIB=$CONDA_PREFIX/lib -DLIBRARY_INC=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$PREFIX
make install
