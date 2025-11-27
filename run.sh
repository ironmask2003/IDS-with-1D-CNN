#!/bin/zsh

REQ_FILE="requirements.txt"
VENV_DIR=".venv"

# Crea la venv se non esiste
if [ ! -d "$VENV_DIR" ]; then
    echo "Creo la virtualenv..."
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install --upgrade pip
    pip install -r $REQ_FILE
else
    echo "Virtualenv già presente."
    source $VENV_DIR/bin/activate
    pip install --upgrade pip
    pip install -r $REQ_FILE
fi

echo "Ambiente pronto!"

# Avvia train o test in base all'input
MODE=""
if [[ "$1" == "train" ]]; then
    MODE="train"
    shift
elif [[ "$1" == "test" ]]; then
    MODE="test"
    shift
else
    echo "Usage: ./run.sh [train|test] [opzioni di main.py]"
    exit 1
fi

if [[ "$MODE" == "train" ]]; then
    python main.py "$@"
elif [[ "$MODE" == "test" ]]; then
    python main.py --test "$@"
fi