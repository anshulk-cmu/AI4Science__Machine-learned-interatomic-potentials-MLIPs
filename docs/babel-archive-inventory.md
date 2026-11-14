# Archive inventory: Bridges-2 project folder moved to Babel

Copy of the inventory written during the move (2026-11-06 to 2026-11-07). The folder itself is not in git. The "Local path" column describes the layout that was copied; those laptop files were deleted after the verification. Current location of everything: `/data/user_data/anshulk/mlip/` on CMU Babel.


## Moved to Babel (2026-11-07)

- All 85 items (89 top-level paths, 125,805 files, 3,053 symbolic links, 395.2 GB) are now in `/data/user_data/anshulk/mlip/` on the CMU Babel cluster. The column "Now on Babel at" gives the path of each item under `mlip/`.
- Verified for every path: sizes, symbolic links, directories and the sha256 of every file on Babel equal the lists made on this laptop and the PSC manifests. `data/omol25/train_4M` and `val` were downloaded again on Babel from the publisher (archives checked against the published size and sha256) and then compared file by file with the laptop copy.
- Hash lists: `manifests\hashes\<item>__<name>.sha256` (laptop) and `manifests\hashes\babel\` (Babel). Ledger: `_tools\move_ledger.jsonl` (copy on Babel: `mlip/logs/ledger.jsonl`). README on Babel: `mlip/README.md` (copy: `manifests\hashes\README_mlip.md`).
- The data files of this laptop copy were deleted on 2026-11-07 after the verification, so the "Local path" column below describes the layout that was copied, not files that are still here. Free space on D: after the deletion: 479 GB.
- Open risk: the backup policy of Babel `/data/user_data` is unknown. PSC still holds the originals and the tars until Bridges-2 goes offline on 9 Nov 2026 05:00 ET.

## Status (state at the time of the copy to this laptop)

- Items verified and extracted: 85 of 85 packed items (85 items planned in total).
- Verified data so far: 395.196 GB in 125805 files and 3053 symbolic links.
- Free space on D: now: 119.559 GB.
- Every tar was checked against the sha256 that was written on PSC in the same pass as the tar. Every extracted item was then compared with the manifest taken on PSC: each file exists with the same size, each symbolic link has the same target, and no extra file exists. After that the tar was deleted from this laptop. One copy only.
- The tars are still on PSC in `/ocean/projects/che260063p/akumar45/localCopy/tars/` (not deleted). So are the manifests, the file lists and the job logs.
- Layout mirrors PSC: `ocean_che260063p\` is `/ocean/projects/che260063p/`, `home_akumar45\` is `/jet/home/akumar45/`.

## Items

| Item | Source on PSC | Local path (under this folder) | GB | Files | Symlinks | sha256 of the tar | Copied | Verified | Now on Babel at | Babel verified |
|---|---|---|---|---|---|---|---|---|---|---|
| akumar45__loose | /ocean/projects/che260063p/akumar45/wandb_sync.log | ocean_che260063p/akumar45/wandb_sync.log | 0.000 | 1 | 0 | tar: eb6f3f685f54cc8076a44b25d92b79fda608eb6088d7ab57e8f6c8c6e8dbd9fa | 2026-11-06 | yes | mlip/archive/ocean_che260063p/akumar45/wandb_sync.log | yes 2026-11-07 |
| akumar45__uma | /ocean/projects/che260063p/akumar45/uma | ocean_che260063p/akumar45/uma | 14.682 | 3 | 0 | tar: f828fd91335ad14579025a789b326192fb59fa7fc08dfa2655002fb5aae6e3d5 | 2026-11-06 | yes | mlip/archive/ocean_che260063p/akumar45/uma | yes 2026-11-07 |
| home__dot_cache__cached_path | /jet/home/akumar45/.cache/cached_path | home_akumar45/.cache/cached_path | 0.102 | 3 | 0 | tar: ef5cbfcc04e726422b8ec92e900d9e0db79a63fe652ed56aff4dbac0496e369c | 2026-11-06 | yes | mlip/archive/home_akumar45/.cache/cached_path | yes 2026-11-07 |
| home__dot_cache__mace | /jet/home/akumar45/.cache/mace | home_akumar45/.cache/mace | 0.044 | 1 | 0 | tar: 068c04aef55c42ec067ea1a37aa06e80caf6cc7995a85094073c3715efccfaeb | 2026-11-05 | yes | mlip/archive/home_akumar45/.cache/mace | yes 2026-11-07 |
| home__mlip_test | /jet/home/akumar45/mlip_test | home_akumar45/mlip_test | 0.000 | 2 | 0 | tar: f7e1085ef68e19b8e5517477c6d45f5b0137a60d9138738053c2ecbbe4f37bcc | 2026-11-05 | yes | mlip/archive/home_akumar45/mlip_test | yes 2026-11-07 |
| shared__cache | /ocean/projects/che260063p/shared/cache | ocean_che260063p/shared/cache | 0.001 | 173 | 0 | tar: 2cdecfbe18e6dea19dc1544eb7281f816276a1d6f4c9d1aa7ea8ad06b6f8c57d | 2026-11-06 | yes | mlip/archive/ocean_che260063p/shared/cache | yes 2026-11-07 |
| shared__data__omol25__pilot | /ocean/projects/che260063p/shared/data/omol25/pilot | ocean_che260063p/shared/data/omol25/pilot | 0.526 | 6 | 0 | tar: 043f1bf22e26dd783852cd2027961f9d186bc21fce5a71f7f2051fa2ad029961 | 2026-11-06 | yes | mlip/data/omol25/pilot | yes 2026-11-07 |
| shared__data__omol25__split | /ocean/projects/che260063p/shared/data/omol25/split | ocean_che260063p/shared/data/omol25/split | 0.148 | 3 | 0 | tar: 2b45b51c46b5045c3af3e185b8698864ac1afee34c2fb395e8971244e7c4cafb | 2026-11-06 | yes | mlip/data/omol25/split | yes 2026-11-07 |
| shared__data__omol25__train_4M | /ocean/projects/che260063p/shared/data/omol25/train_4M | ocean_che260063p/shared/data/omol25/train_4M | 22.797 | 161 | 0 | p01: a42dc5dbdc91dee4885ad61e7c961f1386d1e77f971f485ba281367fc254ef08<br>p02: 0a5a85899600ad0dda3ff02edd570201174ddf2ec86fa526f13c1984d70f9c0e | 2026-11-06 | yes | mlip/data/omol25/train_4M | yes 2026-11-07 |
| shared__data__omol25__val | /ocean/projects/che260063p/shared/data/omol25/val | ocean_che260063p/shared/data/omol25/val | 23.658 | 161 | 0 | p01: 49c120a1df1b225d7881e72c4607f785d06f8742eb7490fe5ae9130ba6da157d<br>p02: 0080281aa5e2240d850712a972d960c95fae141d4af369ab1e1f3ac9a3fe1671 | 2026-11-06 | yes | mlip/data/omol25/val | yes 2026-11-07 |
| shared__dot_conda | /ocean/projects/che260063p/shared/.conda | ocean_che260063p/shared/.conda | 0.000 | 0 | 0 | tar: 9aef2ffba71e1c36f4d8e17c4575fb92b4c53e89358a00a5001d33922816ef29 | 2026-11-06 | yes | mlip/archive/ocean_che260063p/shared/.conda | yes 2026-11-07 |
| shared__dot_pipcache | /ocean/projects/che260063p/shared/.pipcache | ocean_che260063p/shared/.pipcache | 1.628 | 1286 | 0 | tar: 9401ff05e1561bd85ba05c201357ba18fd5e68d47feef97411ed6ca8fcc22e3d | 2026-11-06 | yes | mlip/archive/ocean_che260063p/shared/.pipcache | yes 2026-11-07 |
| shared__envs__loose | /ocean/projects/che260063p/(2 top-level files: see manifests/shared__envs__loose.files.txt) | ocean_che260063p/ (2 files, see manifest) | 0.000 | 2 | 0 | tar: 181b2d0f5ab9ace26ef27bbc97c2bb890033c0ac7a0c1c8891d32a0a2a7104cf | 2026-11-06 | yes | mlip/archive/ocean_che260063p/shared/envs/mace_freeze.txt<br>mlip/archive/ocean_che260063p/shared/envs/mlip_freeze.txt | yes 2026-11-07 |
| shared__envs__mace | /ocean/projects/che260063p/shared/envs/mace | ocean_che260063p/shared/envs/mace | 7.525 | 48384 | 1396 | tar: f0057f910dbf78b1e242399442dba94df94d3847cb7c407433c2738de754df06 | 2026-11-06 | yes | mlip/archive/ocean_che260063p/shared/envs/mace | yes 2026-11-07 |
| shared__envs__mlip | /ocean/projects/che260063p/shared/envs/mlip | ocean_che260063p/shared/envs/mlip | 8.758 | 71473 | 1396 | tar: f64fcbcbbd05b12fce76910d0a941fe076e68b6b6daca6ab89d869621bd98a5c | 2026-11-06 | yes | mlip/archive/ocean_che260063p/shared/envs/mlip | yes 2026-11-07 |
| shared__logs | /ocean/projects/che260063p/shared/logs | ocean_che260063p/shared/logs | 0.001 | 99 | 0 | tar: 322fa3fc3dc5662d079a9ab3a13dbb6817ac91b4a21e4f025f5dd653bd1d891e | 2026-11-06 | yes | mlip/archive/ocean_che260063p/shared/logs | yes 2026-11-07 |
| shared__loose | /ocean/projects/che260063p/(4 top-level files: see manifests/shared__loose.files.txt) | ocean_che260063p/ (4 files, see manifest) | 0.000 | 4 | 0 | tar: f77a3a2ff68330bd5ea21e4234f9082d9e07f7094fa11ff5f3ee012c5edd0d84 | 2026-11-06 | yes | mlip/archive/ocean_che260063p/shared/parse_logs.py<br>mlip/archive/ocean_che260063p/shared/wb_final.py<br>mlip/archive/ocean_che260063p/shared/wb_final_ds.py<br>mlip/archive/ocean_che260063p/shared/wb_final_frozen.py | yes 2026-11-07 |
| shared__repo__AI4Science-MLIPs_untracked | /ocean/projects/che260063p/shared/repo/AI4Science-MLIPs | ocean_che260063p/shared/repo/AI4Science-MLIPs | 0.000 | 19 | 0 | tar: bb860e3f77dca47ce14259362c695bf41b000969fd4465d54c1a7aa221ee0c03 | 2026-11-06 | yes | mlip/archive/ocean_che260063p/shared/repo/AI4Science-MLIPs | yes 2026-11-07 |
| shared__runs__aborted_pilot_dense_s0_49040033 | /ocean/projects/che260063p/shared/runs/aborted_pilot_dense_s0_49040033 | ocean_che260063p/shared/runs/aborted_pilot_dense_s0_49040033 | 0.300 | 20 | 4 | tar: 807a94ac3b05f3d4b8ed503fecefd0425e2739472648f4b4923ae5d0a4e3b902 | 2026-11-06 | yes | mlip/runs/aborted_pilot_dense_s0_49040033 | yes 2026-11-07 |
| shared__runs__aborted_pilot_global_s0_49040034 | /ocean/projects/che260063p/shared/runs/aborted_pilot_global_s0_49040034 | ocean_che260063p/shared/runs/aborted_pilot_global_s0_49040034 | 1.817 | 20 | 4 | tar: c5e7a3409c013b367607f2a3a8c8be9323ca17b1caa10aedc6afd103ebacb131 | 2026-11-06 | yes | mlip/runs/aborted_pilot_global_s0_49040034 | yes 2026-11-07 |
| shared__runs__aborted_pilot_local6_s0_49040032 | /ocean/projects/che260063p/shared/runs/aborted_pilot_local6_s0_49040032 | ocean_che260063p/shared/runs/aborted_pilot_local6_s0_49040032 | 0.911 | 16 | 4 | tar: 3a4100d9f2dd2eeecc353aca20e8ed4f130fdca914757b8685e074e5b77ebd6a | 2026-11-06 | yes | mlip/runs/aborted_pilot_local6_s0_49040032 | yes 2026-11-07 |
| shared__runs__data_lists | /ocean/projects/che260063p/shared/runs/data_lists | ocean_che260063p/shared/runs/data_lists | 0.015 | 237 | 0 | tar: 81b5b9f92f16de4edba180bba86901d9b07ba666300fec6467d6dedce13763b2 | 2026-11-06 | yes | mlip/runs/data_lists | yes 2026-11-07 |
| shared__runs__dense_scaled_s0_49533514 | /ocean/projects/che260063p/shared/runs/dense_scaled_s0_49533514 | ocean_che260063p/shared/runs/dense_scaled_s0_49533514 | 1.737 | 61 | 4 | tar: 5f5ac75ac256d7ef94bf792cab1cc1b2729ae8115cbe19499246461bcfb3a389 | 2026-11-06 | yes | mlip/runs/dense_scaled_s0_49533514 | yes 2026-11-07 |
| shared__runs__dense_scaled_s1_49535801 | /ocean/projects/che260063p/shared/runs/dense_scaled_s1_49535801 | ocean_che260063p/shared/runs/dense_scaled_s1_49535801 | 1.737 | 61 | 4 | tar: 849aa3470339e690d53750a770a8f08e0f72153f0f4ad76774fe455cffacb255 | 2026-11-06 | yes | mlip/runs/dense_scaled_s1_49535801 | yes 2026-11-07 |
| shared__runs__dense_scaled_s2_49535804 | /ocean/projects/che260063p/shared/runs/dense_scaled_s2_49535804 | ocean_che260063p/shared/runs/dense_scaled_s2_49535804 | 1.747 | 61 | 4 | tar: 6c6275014e71b5221b4f2005d6afac10d600f31ebe8d7bdc6af70a04410c5084 | 2026-11-06 | yes | mlip/runs/dense_scaled_s2_49535804 | yes 2026-11-07 |
| shared__runs__dscaledtest_scaled_s0_49532915 | /ocean/projects/che260063p/shared/runs/dscaledtest_scaled_s0_49532915 | ocean_che260063p/shared/runs/dscaledtest_scaled_s0_49532915 | 0.439 | 24 | 4 | tar: bbf4827353fbda3659cc3d15d12b6d8a429600ea0373e44e55befcc40fc68a4a | 2026-11-06 | yes | mlip/runs/dscaledtest_scaled_s0_49532915 | yes 2026-11-07 |
| shared__runs__frozen_global_s0_49444920 | /ocean/projects/che260063p/shared/runs/frozen_global_s0_49444920 | ocean_che260063p/shared/runs/frozen_global_s0_49444920 | 10.103 | 62 | 4 | tar: 8cf6ae3172920787e1ef7a1956d8b9f103bc09d49c0ab048bfac57b8b787436b | 2026-11-06 | yes | mlip/runs/frozen_global_s0_49444920 | yes 2026-11-07 |
| shared__runs__frozen_global_s1_49450911 | /ocean/projects/che260063p/shared/runs/frozen_global_s1_49450911 | ocean_che260063p/shared/runs/frozen_global_s1_49450911 | 10.111 | 62 | 4 | tar: 94351f6626e9d44ba39936c9c8a7d87c8656574acc951a176dcde22bde8e7231 | 2026-11-06 | yes | mlip/runs/frozen_global_s1_49450911 | yes 2026-11-07 |
| shared__runs__frozen_global_s2_49450922 | /ocean/projects/che260063p/shared/runs/frozen_global_s2_49450922 | ocean_che260063p/shared/runs/frozen_global_s2_49450922 | 10.118 | 62 | 4 | tar: c7fec0d3a8297e5fd4e3c7fc12a8c4919e069875c02bb02f654661dd74028042 | 2026-11-06 | yes | mlip/runs/frozen_global_s2_49450922 | yes 2026-11-07 |
| shared__runs__frozentest_global_s0_49442923 | /ocean/projects/che260063p/shared/runs/frozentest_global_s0_49442923 | ocean_che260063p/shared/runs/frozentest_global_s0_49442923 | 2.719 | 25 | 4 | tar: 2f05c19d30cee2cdb531671a84f6fba9dbf7283caa4d7ea5538ae77ded8b6fc8 | 2026-11-06 | yes | mlip/runs/frozentest_global_s0_49442923 | yes 2026-11-07 |
| shared__runs__inference_timing_49443920 | /ocean/projects/che260063p/shared/runs/inference_timing_49443920 | ocean_che260063p/shared/runs/inference_timing_49443920 | 0.002 | 31 | 0 | tar: d16d09e3c1bcf14da24b1326b1804e669b75b90c309de33c08e3484e7eb60478 | 2026-11-06 | yes | mlip/runs/inference_timing_49443920 | yes 2026-11-07 |
| shared__runs__inference_timing_49443929 | /ocean/projects/che260063p/shared/runs/inference_timing_49443929 | ocean_che260063p/shared/runs/inference_timing_49443929 | 0.002 | 31 | 0 | tar: 561b83da7ffb4c2ae13565422ec96aac7acce989a7db9e91d9b0a0b7eb265fb6 | 2026-11-06 | yes | mlip/runs/inference_timing_49443929 | yes 2026-11-07 |
| shared__runs__inference_timing_profile_49445369 | /ocean/projects/che260063p/shared/runs/inference_timing_profile_49445369 | ocean_che260063p/shared/runs/inference_timing_profile_49445369 | 0.002 | 28 | 0 | tar: 93e76970e96d3ff5f8ad6e6b1563a6f831e153caa32badde1ed131538962e1f9 | 2026-11-06 | yes | mlip/runs/inference_timing_profile_49445369 | yes 2026-11-07 |
| shared__runs__long_dense_s0_400000 | /ocean/projects/che260063p/shared/runs/long_dense_s0_400000 | ocean_che260063p/shared/runs/long_dense_s0_400000 | 5.156 | 138 | 4 | tar: e441eae0b28acb8704fafc3a648bf16ec4b871cb829d9b8fba8831454c657198 | 2026-11-06 | yes | mlip/runs/long_dense_s0_400000 | yes 2026-11-07 |
| shared__runs__long_dense_scaled_s0_400000 | /ocean/projects/che260063p/shared/runs/long_dense_scaled_s0_400000 | ocean_che260063p/shared/runs/long_dense_scaled_s0_400000 | 5.162 | 138 | 4 | tar: 55d303f687104591af082ed15405dfc3dfd3d38cfb7d45ded3f704a576bc8c75 | 2026-11-06 | yes | mlip/runs/long_dense_scaled_s0_400000 | yes 2026-11-07 |
| shared__runs__long_element_s0_400000 | /ocean/projects/che260063p/shared/runs/long_element_s0_400000 | ocean_che260063p/shared/runs/long_element_s0_400000 | 16.662 | 86 | 4 | tar: 6d7c057f913beccb90842112c77201805184e3e4a03f44757e5cf5bb9c7d8ecd | 2026-11-06 | yes | mlip/runs/long_element_s0_400000 | yes 2026-11-07 |
| shared__runs__long_global_frozen_s0_400000 | /ocean/projects/che260063p/shared/runs/long_global_frozen_s0_400000 | ocean_che260063p/shared/runs/long_global_frozen_s0_400000 | 24.074 | 118 | 4 | p01: a3e536404ff078c91a8f91b7c2ede152fde32286dec0aac6b0f37fd0d109b336<br>p02: 8389df0d6ef5011d9d38450c5db7bfac68d7cf1f832c1b7ad5565d94fcd0aed7 | 2026-11-06 | yes | mlip/runs/long_global_frozen_s0_400000 | yes 2026-11-07 |
| shared__runs__long_global_s0_400000 | /ocean/projects/che260063p/shared/runs/long_global_s0_400000 | ocean_che260063p/shared/runs/long_global_s0_400000 | 23.164 | 114 | 4 | p01: 224ead641b27ae4d71fe7e6ba18fc9ea9e9d0e0b12c1fb37967af48199b79f8a<br>p02: 31eae145c23c1662cc68bd6d66f06305eba553a217356283165c2efc6697a222 | 2026-11-06 | yes | mlip/runs/long_global_s0_400000 | yes 2026-11-07 |
| shared__runs__long_local12_s0_400000 | /ocean/projects/che260063p/shared/runs/long_local12_s0_400000 | ocean_che260063p/shared/runs/long_local12_s0_400000 | 23.187 | 139 | 4 | p01: c6c83bc9c57350cca685db2e225904eee86cc0f421289dae163d8a17401157b8<br>p02: a732fcd77fb39c413e87ffb9a5d56d1e5e86d79e9fdec2029d73137b25eddccf | 2026-11-06 | yes | mlip/runs/long_local12_s0_400000 | yes 2026-11-07 |
| shared__runs__long_local6_s0_400000 | /ocean/projects/che260063p/shared/runs/long_local6_s0_400000 | ocean_che260063p/shared/runs/long_local6_s0_400000 | 21.340 | 128 | 4 | p01: 0385a6e2a94edd8e8182c4cf9d2207dc493d31db5f20f78689f7dfdac4d6e435<br>p02: a3a534e2c5f24a752c7dad456db00090a77a3ebf5179770207f2c6ad05d00b9b | 2026-11-06 | yes | mlip/runs/long_local6_s0_400000 | yes 2026-11-07 |
| shared__runs__long_local_global_s0_400000 | /ocean/projects/che260063p/shared/runs/long_local_global_s0_400000 | ocean_che260063p/shared/runs/long_local_global_s0_400000 | 23.193 | 138 | 4 | p01: 4b258e762e724016dedf4e711bc70fd52873a776d01affe72b66508dd1e9f9e1<br>p02: 5d873a768b3289869c2185e701efc435765fe8c9753bd30542a6d70431348179 | 2026-11-06 | yes | mlip/runs/long_local_global_s0_400000 | yes 2026-11-07 |
| shared__runs__molecular_eval_49437122 | /ocean/projects/che260063p/shared/runs/molecular_eval_49437122 | ocean_che260063p/shared/runs/molecular_eval_49437122 | 0.001 | 64 | 0 | tar: 93f8669aa4131106db03a7d0cfb5946cfd2f08bc106b1cd09e6567f7dac061c4 | 2026-11-06 | yes | mlip/runs/molecular_eval_49437122 | yes 2026-11-07 |
| shared__runs__pilot_dense_48977565 | /ocean/projects/che260063p/shared/runs/pilot_dense_48977565 | ocean_che260063p/shared/runs/pilot_dense_48977565 | 0.611 | 27 | 4 | tar: 19c77f31a086a2fb434d417b0325a9c8886c7d8a4452a9197708079df4aa4525 | 2026-11-06 | yes | mlip/runs/pilot_dense_48977565 | yes 2026-11-07 |
| shared__runs__pilot_dense_s0_49041937 | /ocean/projects/che260063p/shared/runs/pilot_dense_s0_49041937 | ocean_che260063p/shared/runs/pilot_dense_s0_49041937 | 1.737 | 61 | 4 | tar: bed4df6d4e1f556a76dc092a44bc2d23b76d2216d60828feba8447e2798a43b3 | 2026-11-05 | yes | mlip/runs/pilot_dense_s0_49041937 | yes 2026-11-07 |
| shared__runs__pilot_dense_s1_49041938 | /ocean/projects/che260063p/shared/runs/pilot_dense_s1_49041938 | ocean_che260063p/shared/runs/pilot_dense_s1_49041938 | 1.737 | 61 | 4 | tar: 9b79c22cadc809556d7bcb9424f8a1d514e2faf64a83661a4bcdda53c33017b2 | 2026-11-06 | yes | mlip/runs/pilot_dense_s1_49041938 | yes 2026-11-07 |
| shared__runs__pilot_dense_s2_49041928 | /ocean/projects/che260063p/shared/runs/pilot_dense_s2_49041928 | ocean_che260063p/shared/runs/pilot_dense_s2_49041928 | 1.737 | 61 | 4 | tar: 9737e4d40a7c87fffa4d92754895e31c46faa07d1a3fd64655eaddf86617735f | 2026-11-06 | yes | mlip/runs/pilot_dense_s2_49041928 | yes 2026-11-07 |
| shared__runs__pilot_global_48977567 | /ocean/projects/che260063p/shared/runs/pilot_global_48977567 | ocean_che260063p/shared/runs/pilot_global_48977567 | 3.652 | 27 | 4 | tar: 9e2aac401074eac31225fbc4660560a612bead275794aa34364d6414898f47c7 | 2026-11-06 | yes | mlip/runs/pilot_global_48977567 | yes 2026-11-07 |
| shared__runs__pilot_global_s0_49041939 | /ocean/projects/che260063p/shared/runs/pilot_global_s0_49041939 | ocean_che260063p/shared/runs/pilot_global_s0_49041939 | 10.099 | 61 | 4 | tar: 4fd03be0c9786dfe6698aeafe3c9b64bb72fac570c3a61a0febb9be685c872ba | 2026-11-06 | yes | mlip/runs/pilot_global_s0_49041939 | yes 2026-11-07 |
| shared__runs__pilot_global_s1_49041940 | /ocean/projects/che260063p/shared/runs/pilot_global_s1_49041940 | ocean_che260063p/shared/runs/pilot_global_s1_49041940 | 10.099 | 61 | 4 | tar: aac099ef8a579a1a31ac2c61cb210b329623b99654480ec7fcf5910c1e637f86 | 2026-11-06 | yes | mlip/runs/pilot_global_s1_49041940 | yes 2026-11-07 |
| shared__runs__pilot_global_s2_49041929 | /ocean/projects/che260063p/shared/runs/pilot_global_s2_49041929 | ocean_che260063p/shared/runs/pilot_global_s2_49041929 | 10.110 | 61 | 4 | tar: f251194b1a2e54566e3c870959bde96fe4c99f1480de82841a22e8f0cf089ff9 | 2026-11-06 | yes | mlip/runs/pilot_global_s2_49041929 | yes 2026-11-07 |
| shared__runs__pilot_local6_48977587 | /ocean/projects/che260063p/shared/runs/pilot_local6_48977587 | ocean_che260063p/shared/runs/pilot_local6_48977587 | 3.652 | 27 | 4 | tar: 323da9eb19960a1ac2a2eabe2266bc07083715548a354acc9cebb790463242dc | 2026-11-06 | yes | mlip/runs/pilot_local6_48977587 | yes 2026-11-07 |
| shared__runs__pilot_local6_s0_49041936 | /ocean/projects/che260063p/shared/runs/pilot_local6_s0_49041936 | ocean_che260063p/shared/runs/pilot_local6_s0_49041936 | 10.117 | 61 | 4 | tar: 8af3aae517197a6762a12d96e09625a7d13e0c0b5438ad1917d1620f1ad68679 | 2026-11-06 | yes | mlip/runs/pilot_local6_s0_49041936 | yes 2026-11-07 |
| shared__runs__pilot_local6_s1_49041974 | /ocean/projects/che260063p/shared/runs/pilot_local6_s1_49041974 | ocean_che260063p/shared/runs/pilot_local6_s1_49041974 | 10.110 | 61 | 4 | tar: 47814596e54e93c902e3e28885c62f73d71a3e8d495fd34cbe435a3cb6e03d49 | 2026-11-06 | yes | mlip/runs/pilot_local6_s1_49041974 | yes 2026-11-07 |
| shared__runs__pilot_local6_s2_49041927 | /ocean/projects/che260063p/shared/runs/pilot_local6_s2_49041927 | ocean_che260063p/shared/runs/pilot_local6_s2_49041927 | 10.103 | 61 | 4 | tar: 51e912b7ecdd65ccab628690558048bd6c1487504242d30854b82de048f1e9b2 | 2026-11-06 | yes | mlip/runs/pilot_local6_s2_49041927 | yes 2026-11-07 |
| shared__runs__pilottest_dense_s0_49041064 | /ocean/projects/che260063p/shared/runs/pilottest_dense_s0_49041064 | ocean_che260063p/shared/runs/pilottest_dense_s0_49041064 | 0.439 | 24 | 4 | tar: cd752bd73780cffe62000e6d4e51d8f066c4c19d870b84eb3877cb9c17c2fa03 | 2026-11-06 | yes | mlip/runs/pilottest_dense_s0_49041064 | yes 2026-11-07 |
| shared__runs__pilottest_global_s0_49041065 | /ocean/projects/che260063p/shared/runs/pilottest_global_s0_49041065 | ocean_che260063p/shared/runs/pilottest_global_s0_49041065 | 2.720 | 24 | 4 | tar: 429ab30c614ad2754f8e95bb4646e0a6503d177a6342e579ff13cb85f583ea82 | 2026-11-06 | yes | mlip/runs/pilottest_global_s0_49041065 | yes 2026-11-07 |
| shared__runs__pilottest_local6_s0_49041063 | /ocean/projects/che260063p/shared/runs/pilottest_local6_s0_49041063 | ocean_che260063p/shared/runs/pilottest_local6_s0_49041063 | 2.720 | 24 | 4 | tar: 111e5c27ca48cbf87dd9b307cd6671bc96b3293ebec02ae7d3089ebb3699f561 | 2026-11-06 | yes | mlip/runs/pilottest_local6_s0_49041063 | yes 2026-11-07 |
| shared__runs__prep_long_50003944 | /ocean/projects/che260063p/shared/runs/prep_long_50003944 | ocean_che260063p/shared/runs/prep_long_50003944 | 0.085 | 24 | 0 | tar: 0a9ad14a0382a47cfb9a1508082b23eb3a9027deda2944554d9dfb1dbe97c1f0 | 2026-11-06 | yes | mlip/runs/prep_long_50003944 | yes 2026-11-07 |
| shared__runs__prep_pilot_49039106 | /ocean/projects/che260063p/shared/runs/prep_pilot_49039106 | ocean_che260063p/shared/runs/prep_pilot_49039106 | 0.000 | 5 | 0 | tar: c061db8ba9f43522215bd803e4d4aea76222781b12cdbd9444a3d7f800bd9f9c | 2026-11-06 | yes | mlip/runs/prep_pilot_49039106 | yes 2026-11-07 |
| shared__runs__prep_split_50162814 | /ocean/projects/che260063p/shared/runs/prep_split_50162814 | ocean_che260063p/shared/runs/prep_split_50162814 | 0.000 | 3 | 0 | tar: fcc70d45e0133ec53b472891b4ff0196404d459ebc5f033e3f7b67822edb7712 | 2026-11-06 | yes | mlip/runs/prep_split_50162814 | yes 2026-11-07 |
| shared__runs__prep_split_50164159 | /ocean/projects/che260063p/shared/runs/prep_split_50164159 | ocean_che260063p/shared/runs/prep_split_50164159 | 0.052 | 14 | 0 | tar: 725223a6253a59301f6ac6a151385dee9efc9ebe63662e6ac3ce183cbba4f79c | 2026-11-06 | yes | mlip/runs/prep_split_50164159 | yes 2026-11-07 |
| shared__runs__rerun_48975093 | /ocean/projects/che260063p/shared/runs/rerun_48975093 | ocean_che260063p/shared/runs/rerun_48975093 | 0.000 | 17 | 0 | tar: 7b7492f5eb8c6c534896400c45dbbf08f3879f554538087f35b695c6465833be | 2026-11-06 | yes | mlip/runs/rerun_48975093 | yes 2026-11-07 |
| shared__runs__rerun_48975723 | /ocean/projects/che260063p/shared/runs/rerun_48975723 | ocean_che260063p/shared/runs/rerun_48975723 | 0.000 | 44 | 0 | tar: 7bf1f8ebc5e17dbe000c0b656b7558f6859c1cc5e3ec84d4fb967e7f51e2906d | 2026-11-06 | yes | mlip/runs/rerun_48975723 | yes 2026-11-07 |
| shared__runs__rerun_orb_48975658 | /ocean/projects/che260063p/shared/runs/rerun_orb_48975658 | ocean_che260063p/shared/runs/rerun_orb_48975658 | 0.000 | 2 | 0 | tar: f067558d01d298b5689f8cc794cfee0fb501309d173bf004b536e408eb2556ba | 2026-11-06 | yes | mlip/runs/rerun_orb_48975658 | yes 2026-11-07 |
| shared__runs__rerun_orb_48975738 | /ocean/projects/che260063p/shared/runs/rerun_orb_48975738 | ocean_che260063p/shared/runs/rerun_orb_48975738 | 0.000 | 20 | 0 | tar: 140ea46398c6e2b3822e7976fe382793c04f5c5cbfafc36620ad2b9f2e366b27 | 2026-11-06 | yes | mlip/runs/rerun_orb_48975738 | yes 2026-11-07 |
| shared__runs__rerun_uma_m_49043042 | /ocean/projects/che260063p/shared/runs/rerun_uma_m_49043042 | ocean_che260063p/shared/runs/rerun_uma_m_49043042 | 0.000 | 21 | 0 | tar: dff26887c4a7b83d35b93b19bb311aaad1dfa72cbe83fb1b1262005a4791542a | 2026-11-06 | yes | mlip/runs/rerun_uma_m_49043042 | yes 2026-11-07 |
| shared__runs__resume_test | /ocean/projects/che260063p/shared/runs/resume_test | ocean_che260063p/shared/runs/resume_test | 19.459 | 616 | 48 | tar: 001ee1a8a1d3a2e5c8c0220273d8021dc576d77156068e486dc4d74d90e31c19 | 2026-11-06 | yes | mlip/runs/resume_test | yes 2026-11-07 |
| shared__runs__router_check_49437535 | /ocean/projects/che260063p/shared/runs/router_check_49437535 | ocean_che260063p/shared/runs/router_check_49437535 | 0.007 | 97 | 0 | tar: bfe993e301e9b0583494509b0d0afcd5e4a2f3b338cd24f550e33659b6a2a5c2 | 2026-11-06 | yes | mlip/runs/router_check_49437535 | yes 2026-11-07 |
| shared__runs__router_check_49437538 | /ocean/projects/che260063p/shared/runs/router_check_49437538 | ocean_che260063p/shared/runs/router_check_49437538 | 0.007 | 97 | 0 | tar: 08f846cb1436a306b8a825a2c17c301b345b2b17ea78bb2033845bff39958320 | 2026-11-06 | yes | mlip/runs/router_check_49437538 | yes 2026-11-07 |
| shared__runs__size_consistency_training_49266770 | /ocean/projects/che260063p/shared/runs/size_consistency_training_49266770 | ocean_che260063p/shared/runs/size_consistency_training_49266770 | 0.000 | 33 | 0 | tar: 8e5816e5914ee47de4f156bfa79dc683192a19465e787980c44bfca6a143b687 | 2026-11-06 | yes | mlip/runs/size_consistency_training_49266770 | yes 2026-11-07 |
| shared__runs__smoke64_dense_48978344 | /ocean/projects/che260063p/shared/runs/smoke64_dense_48978344 | ocean_che260063p/shared/runs/smoke64_dense_48978344 | 0.584 | 27 | 4 | tar: 5ab10656cb7ea42320d2718c5a9b8fabfc913773b7496527c1857fb5f4a689aa | 2026-11-06 | yes | mlip/runs/smoke64_dense_48978344 | yes 2026-11-07 |
| shared__runs__smoke64_global_48978349 | /ocean/projects/che260063p/shared/runs/smoke64_global_48978349 | ocean_che260063p/shared/runs/smoke64_global_48978349 | 3.625 | 27 | 4 | tar: d87c965caf5646e5de82d8ed1cc21bddf20b1236926203ddfda4fc87b6dca849 | 2026-11-06 | yes | mlip/runs/smoke64_global_48978349 | yes 2026-11-07 |
| shared__runs__smoke64_local6_48978351 | /ocean/projects/che260063p/shared/runs/smoke64_local6_48978351 | ocean_che260063p/shared/runs/smoke64_local6_48978351 | 3.625 | 27 | 4 | tar: e561f4e0d40ff8aeb569e99936f773c50086acd426af0a58bfaf8e9ff9ec73a8 | 2026-11-06 | yes | mlip/runs/smoke64_local6_48978351 | yes 2026-11-07 |
| shared__runs__smoke_48966949 | /ocean/projects/che260063p/shared/runs/smoke_48966949 | ocean_che260063p/shared/runs/smoke_48966949 | 3.958 | 48 | 12 | tar: 2f9fe5602c91056814b93f58a94d0eccd798a9389d42240f5caaa4957a903a76 | 2026-11-06 | yes | mlip/runs/smoke_48966949 | yes 2026-11-07 |
| shared__runs__smoke_48967167 | /ocean/projects/che260063p/shared/runs/smoke_48967167 | ocean_che260063p/shared/runs/smoke_48967167 | 0.000 | 5 | 0 | tar: 50bb9e15836319cb87d655b586cdf60b136ba9a5d3725c9c056fbd2e2883029e | 2026-11-06 | yes | mlip/runs/smoke_48967167 | yes 2026-11-07 |
| shared__runs__smoke_48967286 | /ocean/projects/che260063p/shared/runs/smoke_48967286 | ocean_che260063p/shared/runs/smoke_48967286 | 3.958 | 57 | 12 | tar: a2350214c490ad1280d90b835a96028701bcef6d6cec893dbad945a70d1c93f5 | 2026-11-06 | yes | mlip/runs/smoke_48967286 | yes 2026-11-07 |
| shared__runs__smoke_48967683 | /ocean/projects/che260063p/shared/runs/smoke_48967683 | ocean_che260063p/shared/runs/smoke_48967683 | 4.263 | 69 | 16 | tar: b77dfbc0caa291c5ad46cfaf2e99ddf0a61976901a6ba3e94f9bc21d2032aab1 | 2026-11-06 | yes | mlip/runs/smoke_48967683 | yes 2026-11-07 |
| shared__runs__smoke_48967685 | /ocean/projects/che260063p/shared/runs/smoke_48967685 | ocean_che260063p/shared/runs/smoke_48967685 | 2.739 | 32 | 8 | tar: 40fd0f4812ee7d2e17c69b854368f3e47c62613c2e48499de53d73c0efc20a0c | 2026-11-06 | yes | mlip/runs/smoke_48967685 | yes 2026-11-07 |
| shared__runs__smoke_integration_50116336 | /ocean/projects/che260063p/shared/runs/smoke_integration_50116336 | ocean_che260063p/shared/runs/smoke_integration_50116336 | 1.462 | 66 | 5 | tar: 22fcc3bc94a7fd32273b7f4e8a5eba20c5493a791a5df42618baeb39ca98dc6c | 2026-11-06 | yes | mlip/runs/smoke_integration_50116336 | yes 2026-11-07 |
| shared__runs__smokeint_element_s0_50116348 | /ocean/projects/che260063p/shared/runs/smokeint_element_s0_50116348 | ocean_che260063p/shared/runs/smokeint_element_s0_50116348 | 2.720 | 24 | 4 | tar: f91443b2640ff059a99fddd9a7844ea06586337704214dab599ac6b8ddbe9c0d | 2026-11-06 | yes | mlip/runs/smokeint_element_s0_50116348 | yes 2026-11-07 |
| shared__runs__smokeint_local12_s0_50116348 | /ocean/projects/che260063p/shared/runs/smokeint_local12_s0_50116348 | ocean_che260063p/shared/runs/smokeint_local12_s0_50116348 | 2.720 | 24 | 4 | tar: a4e49cc15fe591cc11ea89b329723b287a5262bb687dbd8a3fe5ca5bcd791a9b | 2026-11-06 | yes | mlip/runs/smokeint_local12_s0_50116348 | yes 2026-11-07 |
| shared__runs__smokeint_local_global_s0_50116348 | /ocean/projects/che260063p/shared/runs/smokeint_local_global_s0_50116348 | ocean_che260063p/shared/runs/smokeint_local_global_s0_50116348 | 2.720 | 24 | 4 | tar: 9838c5f93dc19c30bd29cadd1cdf2c8f81a4016f24c98151eeb99c488da8a5a5 | 2026-11-06 | yes | mlip/runs/smokeint_local_global_s0_50116348 | yes 2026-11-07 |
| shared__runs__time_eval_50163685 | /ocean/projects/che260063p/shared/runs/time_eval_50163685 | ocean_che260063p/shared/runs/time_eval_50163685 | 0.000 | 2 | 0 | tar: ec0a902baf57a9e8f66d631c7eefef83dded066fda835f9f614fe5c566b96e7c | 2026-11-06 | yes | mlip/runs/time_eval_50163685 | yes 2026-11-07 |
| shared__slurm | /ocean/projects/che260063p/shared/slurm | ocean_che260063p/shared/slurm | 0.000 | 33 | 0 | tar: d43cd65a80cf6bf73011f8051f42ce21ee7cdc9edf8af551eb1aa475952bed59 | 2026-11-06 | yes | mlip/archive/ocean_che260063p/shared/slurm | yes 2026-11-07 |
| shared__tmp | /ocean/projects/che260063p/shared/tmp | ocean_che260063p/shared/tmp | 0.000 | 0 | 0 | tar: e72e6d6e10adc738ee266893657afc5df20d2f77625172a6d6f1366ce48ca042 | 2026-11-06 | yes | mlip/archive/ocean_che260063p/shared/tmp | yes 2026-11-07 |

## What was done on PSC (nothing existing was changed)

- New folder `/ocean/projects/che260063p/akumar45/localCopy/` holds the tars (`tars/`, one `.sha256` per tar), the file lists (`lists/`), the manifests, the plans (`plans/`), the scripts (`jobs/`) and the job logs (`logs/`). It was not deleted.
- Slurm jobs (account che260063p, partition RM-shared, 8 cores): lc_plan 50289365 (12 min 29 s), lc_pack 50290686 (21 min 0 s), lc_ddp 50290735 (8 min 6 s, started after the three DDP training jobs had ended). Together about 5.5 core-hours, so about 5.5 SU.
- Tars are not compressed. Hard links would have been stored as copies; there were none.

## Not copied from PSC

| What | Size | Why |
|---|---|---|
| /ocean/projects/che260063p/shared/data/omol25/train_4M.tar.gz | 19,983,081,456 bytes | The downloaded source of the extracted folder `data/omol25/train_4M`, which is copied. Same data, so a second copy would be a duplicate. sha256 1924f4f50128344cef731069b409757192a83eacdb47e0d0169efc3138ff9688. Stays on PSC. |
| /ocean/projects/che260063p/shared/data/omol25/val.tar.gz | 21,293,980,744 bytes | The downloaded source of the extracted folder `data/omol25/val`, which is copied. sha256 bc2f2aa70459ed029217f188d1e8ab0d90abbb90065235f6414d79246bad1ab8. Stays on PSC. |
| /ocean/projects/che260063p/shared/repo/AI4Science-MLIPs (393 tracked files) | small | Identical (sha256 of every file) to git commit 54eecfe of D:\CMU-Projects-Class\AI4Science-MLIPs. Only the 19 files that are not tracked were copied (item `shared__repo__AI4Science-MLIPs_untracked`). |
| tornado/test/test.key inside envs/mace and envs/mlip | 2 files | A test private key shipped with the Python package tornado. Skipped by the rule not to copy private keys. Not project data. |
| /ocean/projects/che260063p/pbhattad | not readable by akumar45 (mode 700, other user) | Not our files. Size could not be measured. The group quota showed 490.35 GiB used in total on 2026-11-05. |
| /ocean/projects/che260063p/yxia13 | not readable by akumar45 (mode 700, other user) | Not our files. Size could not be measured. The group quota showed 490.35 GiB used in total on 2026-11-05. |
| Pragati's unrelated culture files | not found by name under shared/ or akumar45/ | Not ours. |

### $HOME of akumar45 (/jet/home/akumar45), whole top level

| Name | Size in bytes | Copied | Reason |
|---|---|---|---|
| .vscode-server | 4743813843 | no | VS Code server files, not project data |
| .conda | 915404928 | no | shared conda package cache of all projects |
| .local | 515705729 | no | user-site Python packages, regenerable |
| .cache | 172417246 | no | other caches (pip, conda, torch kernels, matplotlib, copilot); only mace and cached_path are MLIP model files and were copied |
| optout_verification.tgz | 96887639 | no | other project |
| .claude | 20556144 | no | Claude Code login and history on PSC (credentials) |
| nltk_data | 13386063 | no | other project |
| adb_pii | 9047876 | no | other project |
| .nv | 1846202 | no | driver cache |
| .ipynb_checkpoints | 1811236 | no | notebook cruft, other project |
| .lmod.d | 478980 | no | module system settings |
| .ipython | 385395 | no | other project |
| .dotnet | 258301 | no | tool cache |
| survey_domains_v3_addendum_20260813.csv | 193193 | no | other project (August 2026) |
| ondemand | 174642 | no | OnDemand settings |
| .claude.json | 43829 | no | Claude Code login (credentials) |
| .bash_history | 21386 | no | shell history (may hold secrets) |
| .copilot | 21142 | no | tool login data |
| .config | 10106 | no | tool settings |
| .ssh | 2143 | no | SSH keys (credentials) |
| testAllocation_46009058.out | 1802 | no | allocation test, not MLIP |
| sc.txt | 1725 | no | other project (September 1) |
| ssids_ch.txt | 1725 | no | other project |
| d2.py | 1637 | no | other project (September 1) |
| dupcheck.py | 1637 | no | other project (September 1) |
| dd.py | 1536 | no | other project (September 1) |
| dup2.py | 1536 | no | other project (September 1) |
| mlip_test | 1109 | yes | MLIP test scripts |
| .bashrc | 835 | no | shell settings |
| ssids.txt | 678 | no | other project |
| ssids2.txt | 659 | no | other project |
| .zshrc | 658 | no | shell settings |
| .partitions.out | 489 | no | PSC output |
| testAllocation.sbatch | 459 | no | allocation test, not MLIP |
| .claude.json.backup | 436 | no | Claude Code login (credentials) |
| .emacs | 334 | no | shell settings |
| .wget-hsts | 314 | no | tool cache |
| .kshrc | 172 | no | shell settings |
| .condarc | 143 | no | conda settings |
| .bash_profile | 141 | no | shell settings |
| .charge_ids.out | 70 | no | PSC output |
| .bash_logout | 18 | no | shell settings |
| .esd_auth | 16 | no | login data |
| .mozilla | 0 | no | browser data (empty) |
| tmp_ondemand_ocean_cis210027p_symlink | 0 | no | symlink to another allocation |
| .cache/mace | 44422970 | yes | MLIP model download (MACE foundation model, cached_path download) |
| .cache/cached_path | 102097978 | yes | MLIP model download (MACE foundation model, cached_path download) |

## Renamed files (Windows cannot hold two names that differ only in upper and lower case)

Only the terminfo folders of the two conda environments are affected. The first name in sorted order keeps its name, every other name of a colliding group got the suffix ~CASEn. The old and new names are in `manifests\<item>.renamed.txt`. Symbolic link targets were not changed, so a link that pointed into a renamed folder now points to the other folder of the group. This only affects terminfo files, which nothing here runs.

- shared__envs__mace: 11 names renamed (a -> a~CASE1, e -> e~CASE2, l -> l~CASE3, ...).
- shared__envs__mlip: 11 names renamed (a -> a~CASE1, e -> e~CASE2, l -> l~CASE3, ...).

## Long runs: state at copy time

- long_dense_s0_400000: 5.156 GB, 138 files, complete checkpoints up to step 300000 (end of segment 1, from its summary.txt).
- long_dense_scaled_s0_400000: 5.162 GB, 138 files, complete checkpoints up to step 300000 (end of segment 1, from its summary.txt).
- long_element_s0_400000: 16.662 GB, 86 files, complete checkpoints up to step 170000 (end of segment 1, from its summary.txt).
- long_global_frozen_s0_400000: 24.074 GB, 118 files, complete checkpoints up to step 250000 (end of segment 1, from its summary.txt).
- long_global_s0_400000: 23.164 GB, 114 files, complete checkpoints up to step 240000 (end of segment 1, from its summary.txt).
- long_local12_s0_400000: 23.187 GB, 139 files, complete checkpoints up to step 240000 (end of segment 1, from its summary.txt).
- long_local6_s0_400000: 21.340 GB, 128 files, complete checkpoints up to step 220000 (end of segment 1, from its summary.txt).
- long_local_global_s0_400000: 23.193 GB, 138 files, complete checkpoints up to step 240000 (end of segment 1, from its summary.txt).

The segment 1 end lines are in each run's `summary.txt`. The next segments were still pending on PSC at the time of the copy, so these folders are the final state of segment 1 and not a mid-training snapshot.

## How to find things

- Runs: `ocean_che260063p\shared\runs\<run name>\`. The run name is the job name plus the Slurm job id. Inside: `settings.txt`, `summary.txt`, `logs\`, and `train\<id>\checkpoints\step_N\` for the checkpoints.
- The eight long runs are `long_*_s0_400000`. A checkpoint folder `step_N` is complete when it holds `.metadata`, `resume.yaml`, `inference_ckpt.pt` and the `__*_0.distcp` files.
- Training data: `ocean_che260063p\shared\data\omol25\{train_4M,val,split,pilot}`.
- Conda environments: `ocean_che260063p\shared\envs\{mlip,mace}` with `mlip_freeze.txt` and `mace_freeze.txt` next to them. They cannot run on Windows. Rebuild them from the freeze files if needed.
- UMA model weights: `ocean_che260063p\akumar45\uma\*.pt`.
- Job scripts: `ocean_che260063p\shared\slurm\`. Job logs: `ocean_che260063p\shared\logs\`. W&B offline runs are inside each run folder under `logs\wandb\`.
- Code that is not in git: `ocean_che260063p\shared\repo\AI4Science-MLIPs\` (untracked files only). Everything else of the code is in the git repository D:\CMU-Projects-Class\AI4Science-MLIPs.
- `manifests\<item>.files.txt`: size in bytes and path of every file, taken on PSC. `.links.txt`: symbolic links as `path -> target`. `.dirs.txt`: directories. `.excluded.txt`: files that were not copied and why. `.renamed.txt`: names that had to change for Windows (old name, tab, new name).
- `manifests\_plans\<item>.json`: the full record of an item (sizes, parts, sha256, renamed names, problems).
- `_tools\`: the scripts that made and checked this archive. `manifests\_meta\transfer_log.jsonl`: one line per transfer step (scp, sha256, extraction, verification).
- Symbolic links were kept as symbolic links. Their targets are the PSC paths written in the manifests. W&B `latest-run` links and a link to `/ocean/projects/che260063p/shared/cache/wandb/...` therefore point to PSC paths that do not exist on this laptop.

