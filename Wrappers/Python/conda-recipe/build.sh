mkdir ${SRC_DIR}/ccpi
cp -r "${RECIPE_DIR}/../../../" ${SRC_DIR}/ccpi

cd ${SRC_DIR}/ccpi/Wrappers/Python
$PYTHON setup.py install
