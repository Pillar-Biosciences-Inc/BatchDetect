echo '#############################################'
echo '#############################################'
echo '##           Gaussian Likelihood           ##'
echo '#############################################'
echo '#############################################'
python Figure2Creation_LMR.py \
  --pickle Likelihoods.p \
  --pickle LogLikelihoods2.p \
  --pvals-out Computed_pvals2_lmr_gaussian.txt \
  --figure-out Figure2_synthetic_lmr_gaussian.pdf \
  --cd gaussian

echo '#############################################'
echo '#############################################'
echo '##          Hypersecant Likelihood         ##'
echo '#############################################'
echo '#############################################'
python Figure2Creation_LMR.py \
  --pickle Likelihoods.p \
  --pickle LogLikelihoods2.p \
  --pvals-out Computed_pvals2_lmr_hs.txt \
  --figure-out Figure2_synthetic_lmr_hs.pdf \
  --cd hypsecant

echo '###########################################'
echo '###########################################'
echo '##          Student-t Likelihood         ##'
echo '###########################################'
echo '###########################################'
python Figure2Creation_LMR.py \
  --pickle Likelihoods.p \
  --pickle LogLikelihoods2.p \
  --pvals-out Computed_pvals2_lmr_st.txt \
  --figure-out Figure2_synthetic_lmr_st.pdf \
  --cd student_t

echo '#########################################'
echo '#########################################'
echo '##          Laplace Likelihood         ##'
echo '#########################################'
echo '#########################################'
python Figure2Creation_LMR.py \
  --pickle Likelihoods.p \
  --pickle LogLikelihoods2.p \
  --pvals-out Computed_pvals2_lmr_laplace.txt \
  --figure-out Figure2_synthetic_lmr_laplace.pdf \
  --cd laplace

echo '#########################################'
echo '#########################################'
echo '##          Gennorm Likelihood         ##'
echo '#########################################'
echo '#########################################'
python Figure2Creation_LMR.py \
  --pickle Likelihoods.p \
  --pickle LogLikelihoods2.p \
  --pvals-out Computed_pvals2_lmr_gn.txt \
  --figure-out Figure2_synthetic_lmr_gn.pdf \
  --cd gennorm

