echo '#############################################'
echo '#############################################'
echo '##           Gaussian Likelihood           ##'
echo '#############################################'
echo '#############################################'
python Figure2Creation_bootstrap.py \
  --pickle Synthetic_Log_Likelihoods.p \
  --pvals-out Computed_pvals2_boot_gaussian.txt \
  --figure-out Figure2_synthetic_boot_gaussian.pdf \
  --component-distribution gaussian
