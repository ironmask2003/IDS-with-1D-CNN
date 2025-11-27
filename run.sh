#!/bin/zsh


# Colori ANSI
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Funzione per log con cornice
log_box() {
    local color="$1"
    local message="$2"
    local len=${#message}
    local border=""
    for ((i=0; i<$len+4; i++)); do border+="="; done
    echo -e "${color}${border}\n| ${message} |\n${border}${NC}"
}

REQ_FILE="requirements.txt"
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    log_box "$YELLOW" "Creo la virtualenv..."
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    pip install --upgrade pip > /dev/null 2>&1
else
    log_box "$BLUE" "Virtualenv già presente."
    source $VENV_DIR/bin/activate
    pip install --upgrade pip > /dev/null 2>&1
fi

# Installa solo i pacchetti mancanti e mostra log solo per quelli
while read -r pkg; do
    pkg_name=$(echo $pkg | cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1)
    if ! pip show $pkg_name > /dev/null 2>&1; then
        log_box "$YELLOW" "Installazione: $pkg"
        pip install $pkg > /dev/null 2>&1
    fi
done < $REQ_FILE


log_box "$GREEN" "Ambiente pronto!"
echo "" # Riga vuota per separazione visiva
echo "" # Riga vuota per separazione visiva

# Avvia train o test in base all'input
MODE=""

if [[ "$1" == "train" ]]; then
    MODE="train"
    shift
elif [[ "$1" == "test" ]]; then
    MODE="test"
    shift
else
    log_box "$RED" "Usage: ./run.sh [train|test] [opzioni di main.py]"
    exit 1
fi

if [[ "$MODE" == "train" ]]; then
    # Rettangolo con rotellina animata all'interno
    message="Avvio training..."
    final_message="Training completato!"
    # Calcola la lunghezza massima tra la riga animata (con rotellina) e la riga finale
    anim_sample="$message /"
    final_sample="$final_message"
    maxlen=${#anim_sample}
    if [ ${#final_sample} -gt $maxlen ]; then
        maxlen=${#final_sample}
    fi
    maxlen=$(( maxlen + 4 )) # | + spazio + testo + spazio + |
    border=""
    for ((i=0; i<$maxlen; i++)); do border+="="; done
    (
        while true; do
            for c in "|" "/" "-" "\\"; do
                echo -ne "\033[2A" # Vai su di 2 righe
                # Riga animata centrata
                anim_text="$message $c"
                pad_len=$(( $maxlen - 4 - ${#anim_text} ))
                left_pad=$(( pad_len / 2 ))
                right_pad=$(( pad_len - left_pad ))
                anim_line="| "
                for ((j=0; j<$left_pad; j++)); do anim_line+=" "; done
                anim_line+="$anim_text"
                for ((j=0; j<$right_pad; j++)); do anim_line+=" "; done
                anim_line+=" |"
                printf "${CYAN}%s\n%s\n%s${NC}" "$border" "$anim_line" "$border"
                sleep 0.2
            done
        done
    ) &
    SPIN_PID=$!
    python main.py "$@" > /dev/null 2>&1
    kill $SPIN_PID
    wait $SPIN_PID 2>/dev/null
    # Stampa rettangolo finale centrato
    pad_len=$(( $maxlen - 4 - ${#final_message} ))
    left_pad=$(( pad_len / 2 ))
    right_pad=$(( pad_len - left_pad ))
    final_line="| "
    for ((j=0; j<$left_pad; j++)); do final_line+=" "; done
    final_line+="$final_message"
    for ((j=0; j<$right_pad; j++)); do final_line+=" "; done
    final_line+=" |"
    echo -ne "\033[2A" # Vai su di 2 righe
    printf "${GREEN}%s\n%s\n%s${NC}\n" "$border" "$final_line" "$border"
elif [[ "$MODE" == "test" ]]; then
    log_box "$CYAN" "Avvio test..."
    python main.py --test "$@" > /dev/null 2>&1
fi