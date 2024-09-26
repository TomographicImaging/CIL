ROBOCOPY /E "%RECIPE_DIR%\.." "%SRC_DIR%" /XD .git /XD Wrappers\Python\build

set SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CIL=%PKG_VERSION%
if not "%GIT_DESCRIBE_NUMBER%"=="0" (
    set SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CIL=%PKG_VERSION%.dev%GIT_DESCRIBE_NUMBER%+%GIT_DESCRIBE_HASH%
)

cmake "%RECIPE_DIR%\.." -G "NMake Makefiles" -DPython_ROOT_DIR="%BUILD_PREFIX%" ^
  -DCONDA_BUILD=ON ^
  -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
  -DCMAKE_INSTALL_PREFIX=%PREFIX% || exit 1
cmake --build . --target install --config RelWithDebInfo || exit 1
