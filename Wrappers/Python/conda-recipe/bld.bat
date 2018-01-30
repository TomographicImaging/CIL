IF NOT DEFINED CIL_VERSION (
ECHO CIL_VERSION Not Defined.
exit 1
)

ROBOCOPY /E "%RECIPE_DIR%\.." "%SRC_DIR%"

%PYTHON% setup.py build_py
%PYTHON% setup.py install
if errorlevel 1 exit 1
