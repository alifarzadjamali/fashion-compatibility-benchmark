#!/usr/bin/env bash
# Phase 2b (deviation D2): rebuild the IQON preparation catalog wiped with the node, using the
# same scripts as README "Construct the protocols", then extract the IQON v2 images.
# Counts are compared against the committed audits in research/v2/readiness/2026-11-18-g3/g3/.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}
P=$R/v2/iqon_preparation; C=$P/catalog; Z=$R/iqon_audit/IQON3000.zip; PY=.venv/bin/python
mkdir -p $C
echo "=== source audit $(date '+%F %T')"
[ -f $C/items.parquet ] || $PY scripts/audit_iqon3000.py --archive $Z --output $P/source_audit.json --catalog-dir $C
echo "=== image audit $(date '+%F %T')"
[ -f $C/image_audit.parquet ] || $PY scripts/audit_iqon_images.py --archive $Z --items $C/items.parquet \
  --output-parquet $C/image_audit.parquet --output-report $P/image_audit.json
echo "=== resolve variants $(date '+%F %T')"
[ -f $C/image_audit_resolved.parquet ] || $PY scripts/resolve_iqon_image_variants.py --archive $Z --report $P/image_audit.json \
  --image-audit $C/image_audit.parquet --output $C/image_audit_resolved.parquet
ls -la $C
$PY - <<PYEOF
import json
ref='research/v2/readiness/2026-11-18-g3/g3/'
for new,old in [('$P/source_audit.json',ref+'iqon_source_audit.json'),('$P/image_audit.json',ref+'iqon_image_audit.json')]:
    a,b=json.load(open(new)),json.load(open(old))
    keys=[k for k in b if k.endswith('count') or k.endswith('_counts')]
    bad=[k for k in keys if a.get(k)!=b.get(k)]
    print(new, 'MATCH' if not bad else 'MISMATCH '+str(bad), len(keys),'count fields compared')
PYEOF
touch $C/CATALOG_READY
echo "=== extract IQON v2 images $(date '+%F %T')"
[ -f $R/iqon3000_clean/IMAGES_READY ] || { $PY scripts/v2/extract_iqon_images.py --protocol $R/v2/iqon3000_v2_clean/build_a \
  --catalog $C --archive $Z --output $R/iqon3000_clean/images --report $R/iqon3000_clean/extract_report.json \
  && touch $R/iqon3000_clean/IMAGES_READY; }
echo IQONCAT_DONE $(date '+%F %T')
