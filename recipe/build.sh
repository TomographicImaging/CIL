if [ -z "$CIL_VERSION" ]; then
    echo "Need to set CIL_VERSION"
    exit 1
fi
# mkdir ${SRC_DIR}/ccpi
# cp -r "${RECIPE_DIR}/../../../" ${SRC_DIR}/ccpi

# cd ${SRC_DIR}/ccpi/Wrappers/Python
# $PYTHON setup.py install


mkdir ${SRC_DIR}/build_framework
#cp -r "${RECIPE_DIR}/../" ${SRC_DIR}/build_framework

cd ${SRC_DIR}/build_framework
cmake ${RECIPE_DIR}/../ -DCONDA_BUILD=ON -DCMAKE_BUILD_TYPE="Release" -DLIBRARY_LIB=$CONDA_PREFIX/lib -DLIBRARY_INC=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$PREFIX

make install

make install
# $PYTHON setup.py install
