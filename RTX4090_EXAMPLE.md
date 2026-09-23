# Archived RTX 4090 example

Download [rtx4090_typical_example.zip](rtx4090_typical_example.zip) and its [SHA256 checksum](rtx4090_typical_example_SHA256.txt). This package contains the actual archived results for the eight GPU methods in Table 2 and the latency-NMSE plot in Fig. 3. The historical CPU rows of Table 2 are outside this package.

## Recorded setting

- NVIDIA GeForce RTX 4090, PyTorch 2.6.0+cu124, CUDA 12.4, complex64.
- Seed 200; 10 dB SNR; 21 random-phase targets with complex noise.
- Reconstruction grid: 40 by 32; Fourier observations: 20 by 16; M/N = 1/4.
- Each method has 20 device timings and 20 host timings. Recovery quality is measured on one shared scene, not 20 independent scenes.
- Device P50 averages sorted positions 10 and 11. The archived upward-interpolated P95 is the maximum of the 20 timings.

The ZIP contains 16 files under `examples/rtx4090_typical/`: the unchanged source JSON; summary, timing, and quality CSVs; an editable SVG and PDF/PNG exports; a static rebuilding script; and protocol, provenance, and checksum documentation.

## Extract and regenerate archived outputs

Download the ZIP into the repository root and preserve its directory structure when extracting. With Python 3.10 or newer, run from that root:

```bash
python -m zipfile -e rtx4090_typical_example.zip .
python examples/rtx4090_typical/tools/rebuild_archived_outputs.py
```

The second command uses Python's standard library to regenerate `examples/rtx4090_typical/generated/summary.csv` and `examples/rtx4090_typical/generated/latency_nmse_tradeoff.svg`. It does not import or run the numerical solvers. Read the extracted `examples/rtx4090_typical/README.md` for Inkscape export instructions and `PROTOCOL.md` / `PROVENANCE.md` for the full record. The supplied script was checked by static parsing during packaging; it was not executed then.

## Reproduction scope

This is **regeneration of tables and a figure from archived measurements**, not a complete rerun of the original experiment. The available materials do not include the original RTX 4090 runner, the exact seed-200 input generator, or its input arrays. A seed alone does not specify the random-number generator, draw order, scatterer distribution, or noise normalization. Substituting the later RTX 3080 generator would create a different experiment.

The archived literature-baseline source hash matches the released `literature_reproduction/baselines.py`. The archived GGAMP/AFCIFSBL hashes do not match their renamed release counterparts, and FBME core hashes were not recorded in the source JSON. These gaps are documented in the extracted `PROVENANCE.md`; the archive does not claim complete historical source reproducibility.

No solver was rerun or new timing or recovery result produced for this package. It contains no raw radar echo, manuscript, private data, or internal project records. The repository's [licensing notice](NOTICE.md) applies; this example adds no open-source license.
