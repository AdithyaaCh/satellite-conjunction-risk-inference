"""Fill verified Bayesian values and compile the editable research paper."""
import json,subprocess
from pathlib import Path
b=json.loads(Path('outputs/tables/bayesian.json').read_text())['2.0']['conditional_test']
p=Path('reports/results_generated.tex')
p.write_text(f"\\newcommand{{\\BayesCondMAE}}{{{b['mae']:.3f}}}\n\\newcommand{{\\BayesCondN}}{{{b['n']}}}\n\\newcommand{{\\BayesCondCoverage}}{{{100*b['coverage90']:.1f}\\%}}\n")
subprocess.run(['tectonic','--outdir','reports','reports/technical_report.tex'],check=True)
