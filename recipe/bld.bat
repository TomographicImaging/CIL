ROBOCOPY /E "%RECIPE_DIR%\.." "%SRC_DIR%" /XD .git /XD Wrappers\Python\build

set SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CIL="%PKG_VERSION%"
if not "%GIT_DESCRIBE_NUMBER%"=="0" (
    set SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CIL="%PKG_VERSION%.dev%GIT_DESCRIBE_NUMBER%+%GIT_DESCRIBE_HASH%"
)

:: -G "Visual Studio 16 2019" specifies the the generator
:: -T v142 specifies the toolset

cmake -S "%RECIPE_DIR%\.." -B "%SRC_DIR%\build_framework" -G "Visual Studio 16 2019" -T "v142" -DCONDA_BUILD=ON -DCMAKE_BUILD_TYPE=Release -DLIBRARY_LIB=%CONDA_PREFIX%\lib -DLIBRARY_INC=%CONDA_PREFIX% -DCMAKE_INSTALL_PREFIX=%PREFIX%
if errorlevel 1 exit 1

cmake --build "%SRC_DIR%\build_framework" --target install --config Release
if errorlevel 1 exit 1
