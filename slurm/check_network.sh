#!/bin/bash
# Do Prism compute nodes reach Earthdata? If not, run the download stages on the login node
# (slurm/download_on_login.sh) and submit_all.sh --from preprocess.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
for host in https://cmr.earthdata.nasa.gov/search/collections.json?short_name=EMITL1BRAD \
            https://urs.earthdata.nasa.gov https://data.lpdaac.earthdatacloud.nasa.gov https://earth.gov/ghgcenter/api/stac; do
    echo -n "login node  $host : "; curl -s -o /dev/null -m 20 -w "%{http_code}\n" "$host" || echo fail
done
# shellcheck disable=SC2086
srun -p "$CPU_PARTITION" $SBATCH_ACCOUNT_OPT -c 1 -t 5 bash -c '
for host in https://cmr.earthdata.nasa.gov https://data.lpdaac.earthdatacloud.nasa.gov; do
  echo -n "compute ($(hostname)) $host : "; curl -s -o /dev/null -m 20 -w "%{http_code}\n" "$host" || echo fail
done'
