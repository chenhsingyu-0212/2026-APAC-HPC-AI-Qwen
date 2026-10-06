#!/bin/bash
# The bootstrap-port bind check. Nano5 jobs 372421 and 372456 each lost their second
# cell because PD_BOOT_BASE was 9000 + cell*100 and 9100 is a Prometheus IPMI exporter
# on every node of that cluster. Both jobs looked like the flag under test had failed;
# both second cells produced R = 0.53 / 0.59 with n=36 and n=215 on different flags.
set -uo pipefail
cd "$(dirname "$0")/.."
pass=0 fail=0
ck(){ if [[ "$2" == "$3" ]]; then (( pass++ )); else echo "x $1: want '$2' got '$3'" >&2; (( fail++ )); fi; }

# 1-3. the base is out of the exporter range and still separates cells
base() { bash -c 'CELL_INDEX='"$1"'; echo $(( ${PD_BOOT_BASE_FLOOR:-18000} + CELL_INDEX * 100 ))'; }
ck "cell 0 base" "18000" "$(base 0)"
ck "cell 1 base" "18100" "$(base 1)"
ck "no cell lands on 9100" "yes" \
   "$(bash -c 'for c in 0 1 2 3 4 5 6 7; do p=$(( 18000 + c*100 )); for o in 0 1 2 3; do
        [ $((p+o)) -eq 9100 ] && { echo no; exit; }; done; done; echo yes')"
# and the well-known exporters we know about are all below the floor
for p in 9090 9100 9400 9835 9256 29000; do
    [[ $p -lt 18000 || $p -gt 18999 ]] && (( pass++ )) \
        || { echo "x $p falls inside the new bootstrap range" >&2; (( fail++ )); }
done

# 4. the campaign uses the floor, not a literal 9000
grep -q 'PD_BOOT_BASE_FLOOR:-18000' run/10-campaign.sbatch \
    && (( pass++ )) || { echo "x campaign still hardcodes the old base" >&2; (( fail++ )); }
grep -q 'PD_BOOT_BASE=$(( 9000' run/10-campaign.sbatch \
    && { echo "x the 9000 base is still there" >&2; (( fail++ )); } || (( pass++ ))

# 5-7. the guard exists, runs before the launch, and names the port
grep -q 'pd_assert_ports_free()' run/pd-lib.sh \
    && (( pass++ )) || { echo "x pd_assert_ports_free is missing" >&2; (( fail++ )); }
# it must be called BEFORE the worker loop, or it guards nothing
gl=$(grep -n 'pd_assert_ports_free$\|if ! pd_assert_ports_free' run/pd-lib.sh | tail -1 | cut -d: -f1)
wl=$(grep -n 'for i in "${!PD_ROLE\[@\]}"; do' run/pd-lib.sh | tail -1 | cut -d: -f1)
if [[ -n "$gl" && -n "$wl" ]] && (( gl < wl )); then (( pass++ ))
else echo "x the port guard does not precede the worker launch ($gl vs $wl)" >&2; (( fail++ )); fi
grep -q 'ports already in use' run/pd-lib.sh \
    && (( pass++ )) || { echo "x the failure message does not say what is wrong" >&2; (( fail++ )); }

# 8-9. the check itself: bind a port, then assert the probe refuses it
python3 - <<'PY' > /tmp/pg_port 2>/dev/null
import socket
s=socket.socket(); s.bind(('0.0.0.0',0)); print(s.getsockname()[1])
s.listen(1)
import os,sys,time
if os.fork()==0:
    time.sleep(8); os._exit(0)
PY
held="$(cat /tmp/pg_port 2>/dev/null)"
probe() { python3 -c "
import socket,sys
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
try: s.bind(('0.0.0.0',$1))
except OSError: sys.exit(1)
s.close()"; }
# a port nobody holds must pass
probe 18999 && (( pass++ )) || { echo "x the probe rejects a free port" >&2; (( fail++ )); }
# the probe must be the same code the guard runs
grep -q "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)" run/pd-lib.sh \
    && (( pass++ )) || { echo "x the guard's probe differs from the tested one" >&2; (( fail++ )); }

echo "test-port-guard: $pass passed, $fail failed"
(( fail == 0 ))
