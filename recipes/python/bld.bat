IF NOT DEFINED CIL_VERSION (
ECHO CIL_VERSION Not Defined.
exit 1
)

mkdir "%SRC_DIR%\ccpi"
ROBOCOPY /E "%RECIPE_DIR%\..\Wrappers\Python" "%SRC_DIR%\ccpi"
cd ccpi\Wrappers\python

%PYTHON% setup.py install
if errorlevel 1 exit 1
