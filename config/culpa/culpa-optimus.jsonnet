local lib = import '../utils.libsonnet';
local data_type = "culpa";
local latent_dim = 512;
local free_bit = 2.0;
local num_steps = 100000;
local checkout_step = 5000;
local batch_size = 4;
local lr = 1e-5;

{
    "data_dir": "./data/%s" % data_type,
    "model": lib.Optimus(latent_dim, free_bit),
    "trainer": lib.VAETrainer(num_steps, checkout_step, batch_size, lr)
}
