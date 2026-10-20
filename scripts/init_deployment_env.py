"""Create a local Compose env without overwriting credentials or calling services."""
import argparse
from pathlib import Path
import secrets
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".env"))
    args = parser.parse_args()
    template = Path(".env.example").read_text(encoding="utf-8")
    template = template.replace("MYSQL_PASSWORD=change-me",
                                "MYSQL_PASSWORD=" + secrets.token_hex(24))
    template = template.replace("MYSQL_ROOT_PASSWORD=change-root-password",
                                "MYSQL_ROOT_PASSWORD=" + secrets.token_hex(24))
    revision = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, check=False)
    if revision.returncode == 0:
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, check=False)
        suffix = "-dirty" if dirty.returncode == 0 and dirty.stdout.strip() else ""
        template += f"\nSOURCE_REVISION={revision.stdout.strip()}{suffix}\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(template)
    print("Created local deployment configuration. Add provider keys locally before real-model tests.")


if __name__ == "__main__":
    main()
