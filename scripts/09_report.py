"""Build the project PDF from its LaTeX source."""
import subprocess

subprocess.run(['tectonic', '--outdir', 'reports', 'reports/technical_report.tex'], check=True)
