#!/usr/bin/env bash
# Run a BEAST XML on a single machine.
#
# Usage:
#   ./run_local_beast.sh path/to/beast.jar path/to/run.xml [seed] [extra java/beast args...]
#
# Writes output next to the XML: <xml-basename>.log / .ops / stdout, unless
# the XML itself specifies fileName attributes (all XMLs shipped in this
# repo do, and will write into the directory you run this script from).
#
# See environment/BEAST.md for how to build beast.jar from the pinned
# ou-time-series commit.

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <beast.jar> <run.xml> [seed] [extra beast args...]" >&2
  exit 1
fi

beast_jar="$1"
xml_path="$2"
seed="${3:-$(date +%s)}"
shift 3 2>/dev/null || shift "$#"

[ -f "$beast_jar" ] || { echo "BEAST jar not found: $beast_jar" >&2; exit 1; }
[ -f "$xml_path" ] || { echo "XML not found: $xml_path" >&2; exit 1; }

echo "Jar:  $beast_jar"
echo "XML:  $xml_path"
echo "Seed: $seed"

java -jar "$beast_jar" -seed "$seed" -overwrite "$@" "$xml_path"
