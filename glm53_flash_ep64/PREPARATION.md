# EP64 experiment bundle preparation

Scope: package the existing EP64 dummy target and SWA DSpark7 dummy draft,
prepare eight node launchers and an isolated AISBench configuration. Do not
start services, modify framework source, overwrite the original benchmark
configuration, commit, or push.

1. Copy model configurations, quantization metadata, and resolved tokenizer
   files. Preserve source hashes and record any draft metadata adjustment.
2. Extend TP1/DP32 launch parameters to TP1/DP64 with eight local DP8 nodes.
   Centralize placeholder IPs and retain the known experiment settings.
3. Prepare AISBench input262132/output4096/data256/concurrency256/DP64.
   Stage the existing helper files into an isolated runtime directory.
4. Validate JSON, quantization coverage, tokenizer loading, shell syntax,
   all eight dry-run command lines, benchmark dry-run arguments, source
   preservation, and Git visibility. Record remaining hardware limitations.

Existing feature checkout is reused as explicitly requested. Reversible
configuration packaging is checked with parsing and dry runs; no new model
implementation or synthetic unit tests are introduced.
