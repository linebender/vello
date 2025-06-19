# Xtask dev utilities

This package provides the following commands:

## Snapshots

```bash
cargo xtask snaphosts-cpu report  # Creates report for snapshots
cargo xtask snapshots-cpu review  # Interactive test blessing snapshots
cargo xtask snapshots-cpu dead-snaphosts  # Detects dead snapshots
cargo xtask snapshots-cpu size-check  # Size check for snapshots
```

The same works for `snapshots-gpu`

```bash
cargo xtask snaphosts-gpu ...
```


## Comparisons

```bash
cargo xtask comparisons report  # Creates report for comparisons
```
