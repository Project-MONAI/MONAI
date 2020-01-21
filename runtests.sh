#! /bin/bash
set -e
# Test script for running all tests


homedir="$( cd -P "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $homedir

#export PYTHONPATH="$homedir:$PYTHONPATH"

# configuration values
doCoverage=false
doQuickTests=false
doNetTests=false
doDryRun=false
doZooTests=false

# testing command to run
cmd="python"
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
            doCoverage=true
            export QUICKTEST=True
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
        *)
            echo "runtests.sh [--coverage] [--quick] [--net] [--dryrun] [--zoo]"
            exit 1
        ;;
    esac
done


# commands are echoed instead of run in this case
if [ "$doDryRun" = 'true' ]
then
    echo "Dry run commands:"
    cmdprefix="dryrun "

    # create a dry run function which prints the command prepended with spaces for neatness
    function dryrun { echo "  " $* ; }
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
${cmdprefix}${cmd} -m unittest


# network training/inference/eval tests
if [ "$doNetTests" = 'true' ]
then
    for i in examples/*.py
    do
        echo $i
        ${cmdprefix}${cmd} $i
    done
fi


# # run model zoo tests
# if [ "$doZooTests" = 'true' ]
# then
# fi


# report on coverage
if [ "$doCoverage" = 'true' ]
then
    ${cmdprefix}coverage report --omit='*/test/*' --skip-covered -m
fi

