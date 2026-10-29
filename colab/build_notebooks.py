"""Convert the cell-marked .py files into .ipynb notebooks for Google Colab.
Cells start with '# %%' ('# %% [markdown]' for text). Lines starting with '#!' become shell commands,
lines starting with '#from google.colab' become real imports (they are commented so the .py also runs locally)."""
import json
import sys
from pathlib import Path


def convert(src: Path) -> Path:
    cells, cur, kind = [], [], None

    def flush():
        if cur and any(l.strip() for l in cur):
            lines = [l[2:] if kind == "markdown" and l.startswith("# ") else l for l in cur]
            if kind != "markdown":
                lines = [l[1:] if l.startswith("#!") or l.startswith("#from google.colab") else l for l in lines]
            text = "\n".join(lines).strip("\n")
            base = {"cell_type": kind, "metadata": {}, "source": text.splitlines(True)}
            cells.append(base if kind == "markdown" else {**base, "execution_count": None, "outputs": []})

    for line in src.read_text().splitlines():
        if line.startswith("# %%"):
            flush(); cur = []; kind = "markdown" if "[markdown]" in line else "code"
        else:
            cur.append(line)
    flush()
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}}}
    out = src.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1))
    return out


if __name__ == "__main__":
    for f in sorted(Path(__file__).parent.glob("0*.py")):
        print("wrote", convert(f))
