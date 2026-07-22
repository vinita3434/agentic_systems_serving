# SWE-agent-style Agent-Computer Interface (ACI), adapted for a STATELESS
# exec model. Each agent command runs in a fresh `docker exec`, so the
# "currently open file" and window position are persisted to a state file
# ($ACI_STATE) instead of shell env vars. Source this at the top of every
# exec:  source /root/aci.sh; <command>
#
# Commands (mirror SWE-agent semantics):
#   open <path> [line]        open a file, show a window around [line]
#   goto <line>               move the window to <line>
#   scroll_down / scroll_up   move the window by one window-height
#   create <path>             create a new empty file and open it
#   edit <start>:<end> <<'EOF'…EOF   replace lines start..end with heredoc text
#   search_dir <term> [dir]   grep a term across a directory
#   search_file <term> [file] grep a term in a file (default: open file)
#   find_file <name> [dir]    find files by name
#   submit                    (handled by the harness loop; ends the episode)

ACI_STATE="${ACI_STATE:-/root/.aci_state}"
WINDOW="${WINDOW:-100}"

_aci_load() { CURRENT_FILE=""; CURRENT_LINE=1; [ -f "$ACI_STATE" ] && . "$ACI_STATE"; }
_aci_save() { printf 'CURRENT_FILE=%q\nCURRENT_LINE=%q\n' "$CURRENT_FILE" "$CURRENT_LINE" > "$ACI_STATE"; }

# Windowed view of CURRENT_FILE centered on CURRENT_LINE, with line numbers
# and "(N more lines above/below)" markers — the SWE-agent display.
_aci_print() {
    [ -z "$CURRENT_FILE" ] && { echo "No file open. Use: open <path>"; return; }
    python3 - "$CURRENT_FILE" "$CURRENT_LINE" "$WINDOW" <<'PY'
import sys
path, cur, win = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path) as f:
    lines = f.read().splitlines()
n = len(lines)
cur = max(1, min(cur, n if n else 1))
half = win // 2
start = max(1, cur - half)
end = min(n, start + win - 1)
start = max(1, end - win + 1)
print(f"[File: {path} ({n} lines total)]")
if start > 1:
    print(f"({start-1} more lines above)")
for i in range(start, end + 1):
    print(f"{i}:{lines[i-1]}")
if end < n:
    print(f"({n-end} more lines below)")
PY
}

open() {
    _aci_load
    [ -z "$1" ] && { echo "Usage: open <path> [line]"; return 1; }
    [ ! -f "$1" ] && { echo "File $1 not found"; return 1; }
    CURRENT_FILE="$(realpath "$1")"
    CURRENT_LINE="${2:-1}"
    _aci_save; _aci_print
}

goto() {
    _aci_load
    [ -z "$CURRENT_FILE" ] && { echo "No file open. Use: open <path>"; return 1; }
    [ -z "$1" ] && { echo "Usage: goto <line>"; return 1; }
    CURRENT_LINE="$1"; _aci_save; _aci_print
}

scroll_down() { _aci_load; CURRENT_LINE=$((CURRENT_LINE + WINDOW)); _aci_save; _aci_print; }
scroll_up()   { _aci_load; CURRENT_LINE=$((CURRENT_LINE - WINDOW)); _aci_save; _aci_print; }

create() {
    _aci_load
    [ -z "$1" ] && { echo "Usage: create <path>"; return 1; }
    [ -e "$1" ] && { echo "File $1 already exists"; return 1; }
    touch "$1"; CURRENT_FILE="$(realpath "$1")"; CURRENT_LINE=1
    _aci_save; echo "[File: $CURRENT_FILE (0 lines total)]"
}

# edit <start>:<end> with replacement text on stdin (heredoc). Replaces the
# inclusive line range [start,end] in CURRENT_FILE with the piped text.
edit() {
    _aci_load
    [ -z "$CURRENT_FILE" ] && { echo "No file open. Use: open <path>"; return 1; }
    local range="$1"
    case "$range" in
        *:*) : ;;
        *) echo "Usage: edit <start>:<end> then a heredoc of new lines"; return 1;;
    esac
    local start="${range%%:*}" end="${range##*:}"
    local tmp; tmp="$(mktemp)"
    cat > "$tmp"   # replacement text arrives on stdin (heredoc)
    python3 - "$CURRENT_FILE" "$start" "$end" "$tmp" <<'PY'
import sys
path, start, end, tmp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
with open(tmp) as f:
    new = f.read().splitlines()
with open(path) as f:
    lines = f.read().splitlines()
if start < 1 or end > len(lines) or start > end + 1:
    print(f"Invalid range {start}:{end} for {path} ({len(lines)} lines)"); sys.exit(0)
lines[start-1:end] = new
with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"[File: {path} edited: lines {start}-{end} replaced]")
PY
    rm -f "$tmp"
    CURRENT_LINE="$start"; _aci_save; _aci_print
}

search_dir()  { grep -rIn --exclude-dir=.git "${1:?Usage: search_dir <term> [dir]}" "${2:-.}" | head -100; }
search_file() { _aci_load; grep -In "${1:?Usage: search_file <term> [file]}" "${2:-$CURRENT_FILE}" | head -100; }
find_file()   { find "${2:-.}" -name "${1:?Usage: find_file <name> [dir]}" -not -path '*/.git/*' | head -100; }
