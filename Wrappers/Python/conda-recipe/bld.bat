
ROBOCOPY /E "%RECIPE_DIR%\.." "%SRC_DIR%"

%PYTHON% setup.py build_py
%PYTHON% setup.py install
if errorlevel 1 exit 1
