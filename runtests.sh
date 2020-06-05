#! /bin/bash
set -e
# Test script for running all tests


homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$homedir"

export PYTHONPATH="$homedir:$PYTHONPATH"
echo "$PYTHONPATH"

# configuration values
doCoverage=false
doQuickTests=false
doNetTests=false
doDryRun=false
doZooTests=false

doUnitTests=true

doCodeFormatFix=false
doBlackFormat=false
doPytypeFormat=false
doMypyFormat=false
doFlake8Format=false

NUM_PARALLEL=1

# testing command to run
cmd="python3"


# parse arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --coverage)
            doCoverage=true
        ;;
        --quick)
            doQuickTests=true
        ;;
        --net)
            doNetTests=true
        ;;
        --dryrun)
            doDryRun=true
        ;;
        --nounittest*) # allow --nounittest | --nounittests | --nounittesting  etc..
            doUnitTests=false
        ;;
        --zoo)
            doZooTests=true
        ;;
        --codeformat)
          doBlackFormat=true
          doFlakeFormat=true
          doPytypeFormat=true
          doMypyFormat=true
        ;;
        --black)
          doBlackFormat=true
        ;;
        --black-fix)
          doCodeFormatFix=true
          doBlackFormat=true
        ;;
        --flake8)
          doFlake8Format=true
        ;;
        --pytype)
          doPytypeFormat=true
        ;;
        --mypy)
          doMypyFormat=true
        ;;
        -j)
          NUM_PARALLEL=$2
          shift
        ;;
        *)
            echo "ERROR: Incorrect commandline provided"
            echo "Invalid key: $key"
            echo "runtests.sh [--codeformat] [--black] [--black-fix] [--flake8] [--pytype] [--mypy] "
            echo "            [--nounittests] [--coverage] [--quick] [--net] [--dryrun] [--zoo] [-j number]"
            echo "      --codeformat      : shorthand to run all code style and static analysis tests"
            echo "      --black           : Run the \"black\" autoformatting tools as a lint checker"
            echo "      --black-fix       : Apply \"black\" autofix feature"
            echo "      --flake8          : Perform flake8 source code checking"
            echo "      --pytype          : Perform pytype type hint checking"
            echo "      --mypy            : Perform mypy optional static type checker"
            echo "      --nounittests     : skip doing unit testing (i.e. only format lint testers)"
            echo "      --coverage        : Peforms coverage analysis of code for tests run."
            echo "      --quick           : disable long running tests."
            echo "      --net             : perform training/inference/eval integration testing"
            echo "      --dryrun          : display the commands to the screen without running"
            echo "      --zoo             : not yet implmented"
            echo "       -j               : number of parallel jobs to run"
            exit 1
        ;;
    esac
    echo $@
    shift
done

cmdprefix=""
# commands are echoed instead of run in this case
if [ "$doDryRun" = 'true' ]
then
    echo "Dry run commands:"
    cmdprefix="dryrun "

    # create a dry run function which prints the command prepended with spaces for neatness
    function dryrun { echo "  " "$@" ; }
fi

# Unconditionally report on the state of monai
python -c 'import monai; monai.config.print_config()'

# report on code format
if [ "$doBlackFormat" = 'true' ]
then
     set +e  # Disable exit on failure so that diagnostics can be given on failure
     echo "----------------------------------"
     echo "Verifying black formatting checks."
     if [[ ! -f "$(which black)" ]]; then
       # Ensure that the necessary packages for code format testing are installed
       pip install -r requirements-dev.txt
     fi
     black --version

     if [ ${doCodeFormatFix} = 'true' ]; then
       echo "Automaticaly formatting with  black."
       ${cmdprefix}black "$(pwd)"
       black_status=$?
     else
       echo "Verifying black formatting checks."
       ${cmdprefix}black --check "$(pwd)"
       black_status=$?
     fi
     echo "----------------------------------"
     if [ ${black_status} -ne 0 ];
     then
       echo "----------------------------------"
       echo "black code formatting test failed!"
       echo "::: Run"
       echo ":::        black \"$(pwd)\""
       echo "::: to auto fixing formatting errors"
       echo "----------------------------------"
       exit ${black_status}
     else
       echo "*** black code format tests passed. ***"
     fi
     set -e # Enable exit on failure
fi

if [ "$doFlake8Format" = 'true' ]
then
     set +e  # Disable exit on failure so that diagnostics can be given on failure
     echo "-----------------------------------"
     echo "Verifying flake8 formatting checks."
     # Ensure that the necessary packages for code format testing are installed
     if [[ ! -f "$(which flake8)" ]]; then
       pip install -r requirements-dev.txt
     fi
     flake8 --version
     if [ "$doDryRun" = 'true' ]; then
       echo 'MYPYPATH="$(pwd)/monai" flake8 "$(pwd)" --count --statistics'
     else
       MYPYPATH="$(pwd)/monai" flake8 "$(pwd)" --count --statistics
     fi
     flake8_status=$?
     echo "-----------------------------------"
     if [ ${flake8_status} -ne 0 ];
     then
       echo "----------------------------------"
       echo "Formatting test failed!"
       echo "Manually review and fix listed formatting errors"
       exit ${flake8_status}
     else
       echo "*** flake8 code format tests passed. ***"
     fi
     set -e # Enable exit on failure
fi

if [ "$doPytypeFormat" = 'true' ]
then
     set +e  # Disable exit on failure so that diagnostics can be given on failure
     echo "-----------------------------------"
     echo "Verifying pytype typehint checks."
     pytype --version
     # Ensure that the necessary packages for code format testing are installed
     if [[ ! -f "$(which mypy)" ]]; then
       pip install -r requirements-dev.txt
     fi
     ${cmdprefix}pytype -j ${NUM_PARALLEL} --python-version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
     pytype_status=$?
     echo "-----------------------------------"
     if [ ${pytype_status} -ne 0 ];
     then
       echo "----------------------------------"
       echo "Typehinting 'pytype' test failed!"
       echo "Manually review and fix listed typehint errors"
       exit ${pytype_status}
     else
       echo "*** pytype code typehint consistency tests passed. ***"
     fi
     set -e # Enable exit on failure
fi

if [ "$doMypyFormat" = 'true' ]
then
     set +e  # Disable exit on failure so that diagnostics can be given on failure
     echo "-----------------------------------"
     echo "Verifying mypy typehint checks."
     mypy --version
     # Ensure that the necessary packages for code format testing are installed
     if [[ ! -f "$(which mypy)" ]]; then
       pip install -r requirements-dev.txt
     fi
     if [ "$doDryRun" = 'true' ]; then
       echo 'MYPYPATH="$(pwd)/monai" mypy "$(pwd)"'
     else
       MYPYPATH=$(pwd)/monai mypy $(pwd)
     fi
     mypy_status=$?
     echo "-----------------------------------"
     if [ ${mypy_status} -ne 0 ];
     then
       echo "----------------------------------"
       echo "Typehinting 'mypy' test failed!"
       echo "Manually review and fix listed typehint errors"
       exit ${mypy_status}
     else
       echo "*** mypy code typehint consistency tests passed. ***"
     fi
     set -e # Enable exit on failure
fi

# When running --quick, require doCoverage as well and set QUICKTEST environmental
# variable to disable slow unit tests from running.
if [ "$doQuickTests" = 'true' ]
then
    doCoverage=true
    export QUICKTEST=True
fi

# set command and clear previous coverage data
if [ "$doCoverage" = 'true' ]
then
    cmd="coverage run -a --source ."
    ${cmdprefix} coverage erase
fi


# # download test data if needed
# if [ ! -d testing_data ] && [ "$doDryRun" != 'true' ]
# then
# fi


# unit tests
if [ "$doUnitTests" = 'true' ]
then
  python -c 'import torch; print(torch.__version__); print(torch.rand(5,3))'
  ${cmdprefix}${cmd} -m unittest -v
fi


# network training/inference/eval tests
if [ "$doNetTests" = 'true' ]
then
    for i in tests/integration_*.py
    do
        echo "$i"
        ${cmdprefix}${cmd} "$i"
    done
fi

# run model zoo tests
if [ "$doZooTests" = 'true' ]
then
    echo "ERROR:  --zoo options not yet implemented"
    exit 255
fi

# report on coverage
if [ "$doCoverage" = 'true' ]
then
    ${cmdprefix}coverage report --skip-covered -m
fi

