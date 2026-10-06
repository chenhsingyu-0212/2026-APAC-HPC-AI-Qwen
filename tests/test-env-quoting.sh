#!/bin/bash
# `export K=V` written into a launcher is re-parsed by the shell that RUNS it, so a value
# with shell metacharacters has to be quoted at generation time. Job 216211 lost a cell to
# brace expansion eating every quote out of a JSON value -- and inspecting worker.sh
# looked fine, because the file was correct and the shell reading it was the problem.
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || exit 1
fail=0
ck() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fail=1; fi; }

# What pd_write_launcher emits for one K=V line.
emit() { printf 'export %s=%q\n' "${1%%=*}" "${1#*=}"; }
# Generate, then actually execute it -- the bug only appears on execution.
roundtrip() { ( eval "$(emit "$1")"; echo "${!2}" ) ; }

ck "the generator is the one in pd-lib.sh" \
   "$(grep -c "printf 'export %s=%q" run/pd-lib.sh)" "1"

J='SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS={"num_threads":"16","split_batch_size":"256"}'
ck "the JSON survives being re-parsed" \
   "$(roundtrip "$J" SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS)" \
   '{"num_threads":"16","split_batch_size":"256"}'
# The old unquoted form, kept as the negative control: it must NOT round-trip.
ck "the unquoted form does not survive" \
   "$( ( eval "export $J" >/dev/null 2>&1; echo "$SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS" ) \
       | grep -c '"num_threads"' )" "0"

ck "a plain value is unchanged"      "$(roundtrip 'A=1' A)" "1"
ck "a value with a space survives"   "$(roundtrip 'B=/a/b c/d' B)" "/a/b c/d"
ck "a glob is not expanded"          "$(roundtrip 'C=*' C)" "*"
ck "a value containing = keeps it"   "$(roundtrip 'D=k=v=w' D)" "k=v=w"
ck "an empty value stays empty"      "$(roundtrip 'E=' E)" ""
ck "a dollar sign is not expanded"   "$(roundtrip 'F=$HOME' F)" '$HOME'
ck "a semicolon cannot end the statement" "$(roundtrip 'G=a;echo pwned' G)" 'a;echo pwned'

exit $fail
