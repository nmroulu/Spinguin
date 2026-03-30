@ECHO OFF

pushd %~dp0

REM Build script for the Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure that Sphinx is
	echo.installed, then set the SPHINXBUILD environment variable to the full
	echo.path of the 'sphinx-build' executable. Alternatively, add the Sphinx
	echo.directory to PATH.
	echo.
	echo.If Sphinx is not installed, it can be obtained from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
