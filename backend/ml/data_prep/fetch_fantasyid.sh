#!/usr/bin/env bash
# Resumable fetch for FantasyID (2.55 GB) from Zenodo.
#
# This link drops mid-transfer, and curl -C - is the only thing that survives it:
# each attempt resumes from the current byte offset, and --speed-limit aborts a
# stalled stream after 30s instead of hanging on a dead socket. Verified against
# the MD5 published in the Zenodo record, because a resumed download that silently
# lost bytes is worse than one that failed loudly.
set -u
URL="https://zenodo.org/api/records/17063366/files/FantasyID.tgz/content"
OUT="/c/Users/sande/VeriDoc/data/raw/FantasyID.tgz"
EXPECTED_MD5="5a6e608762f640db026ac122b4400a72"

for attempt in $(seq 1 400); do
  curl -L -C - --output "$OUT" \
       --speed-limit 1024 --speed-time 30 \
       --retry 3 --retry-delay 3 --max-time 1800 \
       --silent --show-error "$URL" 2>&1 | tail -1
  rc=$?
  SZ=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
  echo "attempt $attempt: $((SZ/1048576)) MB (curl rc=$rc)"

  if [ "$rc" -eq 0 ] && [ "$SZ" -gt 2500000000 ]; then
    echo "verifying md5..."
    ACTUAL=$(md5sum "$OUT" | cut -d' ' -f1)
    if [ "$ACTUAL" = "$EXPECTED_MD5" ]; then
      echo "COMPLETE and md5 VERIFIED: $ACTUAL"
      exit 0
    fi
    echo "MD5 MISMATCH: got $ACTUAL expected $EXPECTED_MD5"
    exit 2
  fi
  sleep 2
done
echo "gave up after 400 attempts"
exit 1
