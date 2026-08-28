# Research log

Synthetic spectra test:
  I expect the recovered spectra of lam_hat to pull away from the baseline truth at M=640 since this is where our eigenvalues stop since our data only goes up to 640 dimensions, whereas the true spectra will have values in all 1536 dimensions.

  The ratio lam_hat/lam_true should increase up to M=640 since all the variance from the spectra is preserved, it just condensed into fewer dimensions. This early-mid M we will see lam_hat dominate, but as we approach M=640 and go past it this ratio drops to zero, since all eigenvalues of lam_hat are zero after that.

  I predict the trace error will shrink with M and better approximate the true spectra as M increases.

  The participation ratio for D'/D is going to remain small since for finite M D' is always less than D by construction since in the equation (\sum\lam)^2 remains the same whereas \sum\lam^2 will be large for fewer M.
