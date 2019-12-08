# orthonet
Distentangled representations via orthogonal manifolds


## Some practical advice for training an orthogonal model

1. Build a VAE architecture that can fit your data well (with beta_vae = 1)
2. Add a Jacobian-Grammian loss and scan values of beta_ortho. Monitor the dimensionality and orthogonality of the resulting latent spaces with the G_J matrix.
3. If: (a) an orthogonal latent space cannot be obtained without compressing the latent space dimensionality below what is expected, then decrease beta_vae while increasing beta_ortho. Alternatively, if: (b) the dimensionality of the latent space is greater than expected, and compression is desired, try increasing beta_vae.

