ROBOCOPY /E "%RECIPE_DIR%\.." "%SRC_DIR%" /XD .git /XD Wrappers\Python\build

set SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CIL=%PKG_VERSION%
if not "%GIT_DESCRIBE_NUMBER%"=="0" (
    set SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CIL=%PKG_VERSION%.dev%GIT_DESCRIBE_NUMBER%+%GIT_DESCRIBE_HASH%
)
uv pip install . --no-deps
if errorlevel 1 exit 1
