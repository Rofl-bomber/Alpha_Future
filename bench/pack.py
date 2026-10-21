"""Собрать ZIP для платформы: все модули агента в корне архива, без папок.

python bench/pack.py [submission.zip]

В архив попадают только *.py из корня репозитория (solution.py и модули агента).
bench/, tests/, pyproject.toml и uv.lock не кладутся: с ними `uv run` попытался бы ставить
зависимости без сети.
"""

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "submission.zip")
    files = sorted(p for p in ROOT.glob("*.py"))
    assert (ROOT / "solution.py") in files, "нет solution.py"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=p.name)
    print(f"{out}: {len(files)} файлов")
    for p in files:
        print("  ", p.name)


if __name__ == "__main__":
    main()
