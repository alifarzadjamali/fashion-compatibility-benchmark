#!/usr/bin/env bash
# Phase 2b (deviation D2): unpack the dataset archives uploaded from the owner's mirror,
# each only after its SHA-256 matches the corresponding digest below. Waits for each archive.
set -uo pipefail
R=${REPBENCH_DATA_ROOT:-/data/$USER/repbench/data}
wait_sha() {  # path sha
  until [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]; do sleep 60; done
  echo "verified $1 $(date '+%F %T')"
}
# Polyvore (needed first: GPU profiling, near-duplicate audit, construction seeds).
if [ ! -f $R/raw/polyvore_outfits/IMAGES_READY ]; then
  wait_sha $R/raw/polyvore_outfits.zip 03f00b1253dff35c2e9431e84aad9d9a02969f57e57d05975dee38906141f74e
  rm -rf $R/raw/_polyvore_tmp && mkdir -p $R/raw/_polyvore_tmp
  unzip -q $R/raw/polyvore_outfits.zip -d $R/raw/_polyvore_tmp
  mv $R/raw/_polyvore_tmp/polyvore-outfit-dataset/polyvore_outfits $R/raw/polyvore_outfits
  rm -rf $R/raw/_polyvore_tmp
  echo "polyvore images: $(ls $R/raw/polyvore_outfits/images | wc -l)"
  touch $R/raw/polyvore_outfits/IMAGES_READY
fi
for f in "a100_audit/A100.zip 31773883ab6fcf9c0de8182b7561f7319a1d928d542906b8c3c4b8674196f829" \
         "iqon_audit/IQON3000.zip 305660cc09bd941a15096b6efeb536013efe0d534d9cd517419a49cd9ee650fd" \
         "raw/pog/item_data.txt.zip e3ead4e42a04eb4bf74786474983e9b9e255cc13a400130756d08ce2e0883aa3" \
         "raw/pog/outfit_data.txt.zip 9818caf2a1323d2cf7c32a4be7924121bb83b0c7dbcf5d7854b62df94da13a47" \
         "raw/pog/user_data.txt.7z e4f1d8203b9ea52fea86aa8ca850ff273a77f2255cf6c2050b8a38a271be8237"; do
  set -- $f; wait_sha $R/$1 $2
done
# POG images: tar of mirror/ -> raw/pog/mirror.
if [ ! -f $R/raw/pog/IMAGES_READY ]; then
  wait_sha $R/raw/pog/pog_images.tar 8b091ed982d503bec260d1533413eb6292e8e0161eb89dc0dee5aefdc520d2c6
  tar -xf $R/raw/pog/pog_images.tar -C $R/raw/pog
  echo "pog images: $(find $R/raw/pog/mirror -type f | wc -l)"
  touch $R/raw/pog/IMAGES_READY
fi
echo UNPACK_DONE
