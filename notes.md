# Objectives

 - Dataloader yields shuffled, tokenized mazes to training loop
   - Shuffling should be:
     - Order of adj list
     - Order of coord pairs within adj list
     - Order of sections (adjlist, target, origin, NOT path)
   - Shuffling should be controllable with config (not essential for MVP)
 - Training loop performance should be as high as possible
   - Either shuffling a batch needs to be a negligible perf impact, or we precompute it somehow (but then we may have memory concerns)
   - Right now shuffling is done as part of __getitem__, it's not a batch operation, so we're unlikely to make perf worse with these changes)
   - Need to run some benchmarks on different approaches to get clarity here
 - __getitem__ should NOT shuffle (we want dataset[0] == dataset[0])
 - Outside of the training loop, it should be easily possible to get a shuffled or unshuffled tokenized maze (either shuffled adjlist, or shuffled entire tokenized maze)
 - __getitem__ should perhaps return SolvedMazes (probably the most ergonomic option)
 - If possible, remove some duplicate accessor stuff from dataset
 - 

# Approach

 - Dataloader collate_fn looks promising. Operated on yielded batch of samples from Dataset. This seems like a good place for tokenization and shuffling (and would likely be faster as it happens on the batch)


