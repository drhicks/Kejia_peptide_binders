# Protein Diffusion

Codes for running Protein Diffusion.

## Description
Protein Diffusion is a method for structure generation, with or without conditional information (a motif, target etc). It is still in development, but should be able to perform a whole range of protein design challenges.

We will try and keep this repo stable-ish

**Things Diffusion can (in theory) do**
- Motif Scaffolding (a la inpainting, but perhaps from less input)
- Unconditional protein generation
- Symmetric unconditional generation (cyclic, dihedral and tetrahedral symmetric currently implement, more coming)
- Symmetric motif scaffolding (with and without specifying the relative position of the motif in the asymmetric unit)
- Binder design
- Design diversification (sampling around a design)

**Things we are planning on adding**
- Sequence design (currently the sequence on the backbone is basically random)
- Ability to condition on secondary structure/block adjacency information

## Getting started
To get started using inpainting, you will need to clone this git repository.

1. Navigate to a place on digs where you would like to clone this repo
2. `git clone https://git.ipd.uw.edu/jwatson3/RFdiffusion.git` 

## Usage
In this section we will demonstrate how to run diffusion

### The python environment
Any time you're running inpainting, be sure to `source activate XYZ` because the code has dependencies located in that python environment. For now, use this environment:
`/home/dimaio/.conda/envs/SE3nv`

You will need to also pip install a few modules:
`pip install icecream`
`pip install hydra-core --upgrade`
`pip install OmegaConf`

### Running the diffusion script
The actual script you will execute is called `run_inference.py`. There are many ways to run it. It is conceptually similar to how inpainting is run, but now runs with hydra configs, which work a little bit differently to the "hundreds-of-arguments" setup of inpainting.
Hydra configs are a nice way of being able to specify many different options, with sensible defaults drawn *directly* from the model checkpoint, so inference should always, by default, match training.

**Basic execution**
Let's first look at how you would do unconditional design of a protein of length 150aa.
For this, we just need to specify two things:
1. The length of the protein
2. The location where we want to write files to
3. The number of designs we want

```
./run_inference.py contigmap.contigs=[\'150-150\'] inference.output_prefix=test_outputs/test inference.num_designs=10
```

Let's look at this in detail.
Firstly, what is `contigmap.contigs`?
Hydra configs tell the inference script how it should be run. To keep things organised, the config has different sub-configs, one of them being `contigmap`, which pertains to everything related to the contig string (that defines the protein being built).
Take a look at the config file if this isn't clear: `configs/inference/base.yml`
Anything in the config can be overwritten manually from the command line. You could, for example, change how the diffuser works:
```
diffuser.chi_type=wrapped_normal
```
... but don't do this unless you really know what you're doing!!


Now, what does `[\'150-150\']` mean?
To those who have used inpainting, this might look familiar, but a little bit different. Diffusion, in fact, uses the identical 'contig mapper' as inpainting, except that, because we're using hydra, we have to give this to the model in a different way. `'` tokens mean something to hydra, so we have to 'escape' them, with the backslash. The contig string also has to be passed as a single-item in a list, rather than as a string, again for hydra reasons.
The contig string allows you to specify a length range, but here, we just want a protein of 150aa in length, so you just specify [\'150-150\']
This will then run 10 diffusion trajectories, saving the outputs to your specified output folder.

NB the first time you run diffusion, it will take a while 'Calculating IGSO3'. Once it has done this, it'll be cached for future reference though!

**Motif Scaffolding**
Diffusion can be used to scaffold motifs, in a manner akin to inpainting. For some problems, inpainting is probably still better than diffusion, but for others, especially those with only a small motif, we expect diffusion to outperform inpainting.

If we want to scaffold a motif, the input is just like inpainting, but taking into account the specifics of the hydra config input. If we want to scaffold residues 10-25 on chain A a pdb, whereas in inpainting this would be specified with `--contigs 5-15,A10-25,30-40`, in diffusion this would be done with `contigmap.contigs=[\'5-15,A10-25,30-40\']`. If we wanted to ensure the length was always e.g. 55 residues, this can be specified with `contigmap.length=55-55`. You need to obviously also provide a path to your pdb file: `inference.input_pdb=path/to/file.pdb`. It doesn't matter if your input pdb has residues you *don't* want to scaffold - just like inpainting, the contig map defines which residues in the pdb are actually used as the "motif". 

Just like in inpainting, we can specify that we want to inpaint in the presence of a separate chain, this can be done as follows:

```
contigmap.contigs=[\'B1-100,0\ '5-15,A10-25,30-40\']
```
Look at this carefully. Just as with inpainting, the symbol for a chain break is `,0 `. However, the space symbol, just like the `'` symbol, means something in hydra, so we have to again 'escape' it. NOTE, the space is important here. This tells the diffusion model to add a big residue jump (200aa) to the input, so that the model sees the first chain as being on a separate chain to the second. We recognise this contig input, with all the hydra stuff, is a pain, and might change it in future.

Currently, for two-chain inpainting, the output will be all on the same chain. We will update this. However, for now, don't worry - the chains are unrelated to what diffusion sees, and it 'saw' them as two separate chains.

**Binder Design**
Hopefully, it's now obvious how you might make a binder with diffusion! If chain B is your target, then you could do it like this:

```
./run_inference.py contigmap.contigs=[\'B1-100,0\ 100-100\'] inference.output_prefix=test_outputs/binder_test inference.num_designs=10 ppi.binder_len=100
```

However, this probably isn't the best way of making binders. Because diffusion is relatively computationally-intensive, we need to try and make it as fast as possible. Providing the whole of your target, uncropped, is going to make diffusion very slow if your target is big. But, if you crop your target, how can you guarantee the binder will go to part of the surface of the target, and not the backside of the cropped target?

This can be achieved in one of two ways. This README will be updated when we know for sure which is the best way of doing this.
1. With specified `hotspot` residues.
Some models have been trained on complexes, and in this training regime, the model was told (some of) the residues the 'binder' binds to. Therefore, if we know which residues in our target are good residues to be targeting, these can be specified at inference time, like so `ppi.hotspots=[\'A30\',\'A33\',\'A34\']`
2. With a potential. 
The beauty of diffusion is that we can guide the generation process, with 'auxiliary potentials'. These bias the updates at each timestep, towards some prespecified 'goal'. In this case, this goal, or potential, might be 'bind to these residues'. This can be done with the following command:

`potentials.guiding_potentials=[\'type:binder_distance_ReLU,min_dist:8\'] ppi.hotspot_res=[\'B146\',\'B170\',\'B177\']`

This potential pushes the 'binder' towards the specified hotspot residues.

We have to also set how strongly we weight this potential. This is not fully optimised/benchmarked yet, but the following is probably okay:

`potentials.guide_scale=guide_scale=2`

And probably, we don't want to apply the potential the whole way through the trajectory (as once the 'fold' is set, it's probably unhelpful/unneeded).

`potentials.guide_decay='quadratic'`

Currently, linear, quadratic and cubic decays are implemented. This README will be updated as and when we develop 'best' practices.

This is a good time to introduce potentials in more detail. We have already implemented a few, but it is relatively straightforward to add more, if you want to push your designs towards some specified goal. The *only* condition is that, whatever potential you write, it is differentiable. Take a look at `potentials.potentials.py` for examples of the potentials we have implemented so far.

**Using Auxiliary Potentials**
Auxiliary potentials are *great* for guiding the inference process. E.g. whereas in inpainting, we have little/no control over the final shape of an output, in diffusion we can readily force the network to make a well-packed protein.
This is achieved in the updates we make at each step. Let's go a little deeper into how the diffusion process works:
At timestep T (the first step of the reverse-diffusion inference process), we sample noise from a known *prior* distribution. The model then makes a prediction of what the final structure should be, and we use these two states (noise at time T, prediction of the structure at time 0) to back-calculate where t=T-1 would have been. We therefore have a vector pointing from each coordinate at time T, to their corresponding, back-calculated position at time T-1.
But, we want to be able to bias this update, to *push* the trajectory towards some desired state. Well, this can be done by biasing that vector with another vector, which points towards a position where that residue would *reduce* the 'loss' as defined by your potential. E.g. if we want to use the `monomer_ROG` potential, which seeks to minimise the radius of gyration of the final protein, if the models prediction of t=0 is very elongated, each of those distant residues will have a larger gradient when we differentiate the `monomer_ROG` potential w.r.t. their positions. These gradients, along with the corresponding scale, can be combined into a vector, which is then combined with the original update vector to make a "biased update" at that timestep.

The exact parameters of how we apply these potentials matters. If you weight them too strongly, you're not going to end up with a good protein. Too weak, and they'll have no effect. We've explored these potentials in a few different scenarios, and have set sensible defaults, if you want to use them. But, if you feel like they're too weak/strong, or you just fancy exploring, do play with the parameters (in the `potentials` part of the config file). If you develop something useful, or find some much better parameters, let us know! We'd love to incorporate improvements into the pipeline.

Here's an example of another useful potential:
- olig_contacts
This potential seeks to maximise contacts, and can be applied intra- or inter-chain, with different relative weighting. Here are Helen's parameters currently used for oligomer design, which seem sensible:

```
potentials.guiding_potentials=[\"type:olig_contacts,weight_intra:1,weight_inter:0.1\"] potentials.olig_intra_all=True potentials.olig_inter_all=True potentials.guide_scale=2 potentials.guide_decay='quadratic'```

**Generation of Symmetric Oligomers**
Diffusion can directly generate symmetric oligomers. This is done by symmetrising the noise we sample at t=T, and symmetrising the input at every timestep. We have currently implemented:
- Cyclic symmetry
- Dihedral symmetry
- Tetrahedral symmetry

Here's an example:
```
./run_inference.py --config-name symmetry  inference.symmetry=tetrahedral contigmap.contigs=[\"360\"] inference.output_prefix=test_sample/tetrahedral inference.num_designs=1
```

Here, we've specified a different `config` file (with `--config-name symmetry`). Because symmetric diffusion is quite different from the diffusion described above, we packaged a whole load of symmetry-related configs into a new file (see `configs/inference/symmetry.yml`). Using this config file now puts diffusion in `symmetry-mode`.

The symmetry type is then specified with `inference.symmetry=`. Here, we're specifying tetrahedral symmetry, but you could also choose cyclic (e.g. `c4`) or dihedral (e.g. `d2`).

The configmap.contigs length refers to the *total* length of your oligomer. Therefore, it *must* be divisible by *n* chains.

**Symmetric Motif Scaffolding**
We can also combine symmetric diffusion with motif scaffolding, to scaffold motifs symmetrically!
This will be possible in two different ways:
1. With the position of the motif specified w.r.t. the symmetry axis defined
2. With the position the the motif within the asymmetric unit chosen (spatially) by the model.
Currently, only #1 is implemented

These two modes require different input specifications.
1. Here, we want to make sure people know what they're doing, so have made this, purposefully, a little cumbersome. We have defined symmetry axes in `symmetry.py`. If you want to specify the position of a motif within each asymmetric unit w.r.t. the symmetry axes (e.g. if you want to build an oligomer that binds a metal, with the binding site assembled from all of the different chains and with the proper geometry for metal coordination, *you* need to make this input (in e.g. pymol). I.e., to make a tetrahedral motif scaffold, you need to position the four motifs around *our* symmetry axes, such that your input pdb is symmetric.
An example of a input is given below:
```
#TODO DJ to add
```

**A Note on Model Weights**
Because of everything we want diffusion to be able to do, there is not *One Model To Rule Them All*. E.g., if you want to run with secondary structure conditioning, this requires a different model than if you don't. Under the hood, we take care of most of this by default - we parse your input and work out the most appropriate checkpoint.
This is where the config setup is really useful. The exact model checkpoint used at inference contains in it all of the parameters is was trained with, so we can just populate the config file with those values, such that inference runs as designed.
If you do want to specify a different checkpoint (if, for example, we train a new model and you want to test it), you just have to make sure it's compatible with what you're doing. E.g. if you try and give secondary structure features to a model that wasn't trained with them, it'll crash.

**Things you might want to play with at inference time**
For a full list of things that are implemented at inference, see the config file (`configs/inference/base.yml` or `configs/inference/symmetry.yml`). Although you can modify everything, this is not recommended unless you know what you're doing.
Generally, don't change the `model`, `preprocess` or `diffuser` configs. These pertain to how the model was trained, so it's unwise to change how you use the model at inference time.
However, the parameters below are definitely worth exploring:
-inference.final_step: This is when we stop the trajectory. We have seen that you can stop early, and the model is already making a good prediction of the final structure. This speeds up inference.
-diffuser.T: This is the only diffusion parameter you might want to change. This is the total number of steps the model is diffused-to-noise over. The model was trained with 200 timesteps, so in other words, we diffuse the model to a Gaussian distribution over 200 timesteps. But, if we set T to 100, we can just take double the step size, and now noise (or denoise) over 100, bigger steps.
The defaults in this repo are `T=100` and `final_step=25`. This is based on comprehensive benchmarking data. The quality of models, when the designs are big, is slightly lower with `T=100` as compared to `T=200`, but given that you can run twice as many trajectories, you'll get more good outputs per unit time using `T=100`. If you're not getting any good outputs, you might want to try `T=200` though. The `final_step=25` stops trajectories at t=25, and comes at no cost to model quality.

-denoiser.noise_scale_ca: This can be used to reduce the noise used during sampling. The default is 1 (the same noise added at training), but this can be reduced to e.g. 0.5, or even 0. This actually improves the quality of models coming out of diffusion, but at the expense of diversity. If you're not getting any good outputs, or if your problem is very constrained, you could try reducing the noise.

**Conclusion**
RFdiffusion is a rapidly evolving codebase. This means things will change regularly. We'll try and keep everyone updated with what's going on/what improvements have been made. Please let us know if things don't work, or if you discover any cool tricks or develop new methods/potentials/helper scripts etc. Also, let us know what you'd like! We'd love to help you in whatever problems you're working on.

Enjoy!!

