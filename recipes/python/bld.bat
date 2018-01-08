IF NOT DEFINED CIL_VERSION (
ECHO CIL_VERSION Not Defined.
exit 1
)

mkdir "%SRC_DIR%\ccpi"
xcopy /e "%RECIPE_DIR%\..\..\.." "%SRC_DIR%\ccpi"
cd ccpi\Wrappers\python

%PYTHON% setup.py build_ext
if errorlevel 1 exit 1
%PYTHON% setup.py install
if errorlevel 1 exit 1
