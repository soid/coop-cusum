local lib = import '../utils.libsonnet';
local data_type = "culpa";
local latent_dim = 512;
local free_bit = 0.25;
local num_steps = 10000;  #100000;
local checkout_step = 1000;
local batch_size = 128;
local lr = 1e-3;

{
    "data_dir": "./data/%s" % data_type,
    "spm_path": "./data/sentencepiece/culpa4.model",
    "model": lib.BiMeanVAE(latent_dim, free_bit),
    "trainer": lib.VAETrainer(num_steps, checkout_step, batch_size, lr)
}