"""Update generated case-study metrics in the consolidated manuscript source."""
from pathlib import Path


def update_metrics(manuscript: Path, case: int, tex: str) -> None:
    """Replace one marked metrics block while preserving all surrounding prose."""
    start = f'% BEGIN GENERATED CASE STUDY {case} METRICS\n'
    end = f'% END GENERATED CASE STUDY {case} METRICS'
    source = manuscript.read_text()
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f'Expected one inline metrics block for Case Study {case}: {manuscript}')
    before, _, tail = source.partition(start)
    _, _, after = tail.partition(end)
    updated = before + start + tex.rstrip() + '\n' + end + after
    if updated != source:
        manuscript.write_text(updated)
