#!binbash
set -e

# Move to project root (parent directory of this script's directory)
cd $(dirname $0)..

echo === Running all experiments ===

python codeanalysisbatch_scan.py
python codeanalysislocal_pca_analysis.py
python codeanalysiscross_model_analysis.py
python codeanalysishead11_deep_analysis.py
python codeanalysislayer11_analysis.py
python codeanalysisablation_v2.py
python codeanalysiscumulative_ablation.py
python codeanalysisdomain_width_experiment.py
python codeanalysisverify_A1.py

echo === Generating appendix figures ===

python codeutilsplot_figA1.py
python codeutilsplot_figA2.py
python codeutilsplot_figA3.py
python codeutilsplot_figA4.py
python codeutilscrop_fig01_fig02.py

echo === All done ===