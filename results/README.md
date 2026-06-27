# HADROS3 Results Workspace

This directory is the audited workspace for scientific HADROS3 results.

The goal is to keep every important result reproducible and interpretable:

```text
run
catalog
validate
interpret
select figures
preserve provenance
```

Recommended layout:

```text
results/catalog/
results/validation/
results/scans/
results/paper_candidates/
```

Each curated result package should contain:

```text
README.md
config.json
provenance.json
summary.json
figures/
tables/
notes.md
```

The central catalog lives in:

```text
results/catalog/HADROS3_RESULTS_CATALOG.csv
results/catalog/HADROS3_RESULTS_CATALOG.json
```

Use `scripts/results/register_result.py` to register a generated
`output/<run-name>/` directory without modifying the original run outputs.
