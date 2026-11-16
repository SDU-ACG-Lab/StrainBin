import subprocess

def run_command(command):
    """Run a shell command and return the output."""
    result = subprocess.run(
        "set -o pipefail; " + command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        shell=True,
        executable="/bin/bash",
    )
    return result.stdout.strip()
