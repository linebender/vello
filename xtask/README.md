# Xtask dev utilities

This package provides the following commands:

## Snapshots

```bash
cargo xtask snaphosts-cpu report  # Creates report for snapshots
cargo xtask shapshots-cpu review  # Interactive test blessing snapshots
cargo xtask shapshots-cpu dead-snaphosts  # Detects dead snapshots
cargo xtask shapshots-cpu size-check  # Size check for snapshots
```

The same works for `snapshots-gpu`

```bash
cargo xtask snaphosts-gpu ...
```


## Comparisons

```bash
cargo xtask comparisons report  # Creates report for comparisons
```
