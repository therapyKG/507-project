#lang rosette

(require "syntax.rkt")
(require "validate.rkt")

(module+ main
  (printf "Test 1: Standard MNIST (Should pass easily)\n")
  (train
   (model (bits 32)
    (sequence
     (linear 784 128 relu)
     (linear 128 10 softmax)))
   on
   (dataset "MNIST" 60000 (bits 32) 784)
   with ((vram 1024) (batch 64)))

  (printf "\nTest 2: High Res OOM Simulation\n")
  ;; This should fail the requested batch, but tell us what the MAX batch is.
  (train
   (model 
    (bits 16)
    (sequence
     (conv 3 64 3 relu)
     (conv 64 64 3 relu)
     (maxpool 2)
     (flatten)
     (linear 16777216 10)))
   on
   (dataset "HD-Image" 5000 3 1024 1024 (bits 16))
   with ((vram 81920) (batch 128)))
   
   (printf "\nTest 3: Dataset Caching Check\n")
   ;; Small model, Small VRAM, Large Dataset. 
   ;; We want to see if it calculates that we can't cache the whole dataset.
   (train
    (model (bits 32)
     (sequence (linear 10 10)))
    on
    (dataset "BigData" 1000000 (bits 32) 10) ;; ~40MB dataset
    with ((vram 20) (batch 32)))
)