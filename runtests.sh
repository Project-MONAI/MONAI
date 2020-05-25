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
doCodeFormat=false

# testing command to run
cmd="python3"
cmdprefix=""


# parse arguments
for i in "$@"
do
    case $i in
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
        --zoo)
            doZooTests=true
        ;;
        --codeformat)
            doCodeFormat=true

        ;;
        *)
            echo "runtests.sh [--codeformat] [--coverage] [--quick] [--net] [--dryrun] [--zoo]"
            exit 1
        ;;
    esac
done

# report on code format
if [ "$doCodeFormat" = 'true' ]
then
     # Ensure that the necessary packages for code format testing are installed
     if [[ ! -f "$(which flake8)" ]] || [[ ! -f "$(which black)" ]]; then
       pip install -r requirements-dev.txt
     fi

     set +e  # Disable exit on failure so that diagnostics can be given on failure
     echo "----------------------------------"
     echo "Verifying black formatting checks."
     black --check "$(pwd)"
     black_status=$?
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

     echo "-----------------------------------"
     echo "Verifying flake8 formatting checks."
     MYPYPATH="$(pwd)/monai" flake8 "$(pwd)" --count --statistics
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
     set -e # Re-enable exit on failure
fi

# When running --quick, require doCoverage as well and set QUICKTEST environmental
# variable to disable slow unit tests from running.
if [ "$doQuickTests" = 'true' ]
then
    doCoverage=true
    export QUICKTEST=True
fi

# commands are echoed instead of run in this case
if [ "$doDryRun" = 'true' ]
then
    echo "Dry run commands:"
    cmdprefix="dryrun "

    # create a dry run function which prints the command prepended with spaces for neatness
    function dryrun { echo "  " "$@" ; }
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
${cmdprefix}${cmd} -m unittest -v


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

