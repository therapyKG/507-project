#lang rosette

(require "syntax.rkt")
(require "validate.rkt")

(module+ main
  (printf "Test 1: Simple Sequential Pipeline\n")
  (pipeline
   (train (model (bits 32) (sequence (linear 784 128 relu) (linear 128 10))) 
          on (dataset "MNIST-A" 10000 (bits 32) 784))
   (train (model (bits 32) (sequence (linear 784 64 relu) (linear 64 10))) 
          on (dataset "MNIST-B" 5000 (bits 32) 784))
   with ((vram 1024)))

  (printf "\nTest 2: Tight VRAM Constraint (Shared Budget)\n")
  (pipeline
   (train (model (bits 16) (sequence (conv 3 32 3 relu) (maxpool 2) (flatten) (linear 7200 10)))
          on (dataset "CIFAR" 2000 (bits 16) 3 32 32))
   (train (model (bits 32) (sequence (linear 1024 1024 relu) (linear 1024 10)))
          on (dataset "VectorData" 2000 (bits 32) 1024))
   with ((vram 50))) 
  
  (printf "\nTest 3: Impossible Pipeline (OOM)\n")
  (pipeline
   (train (model (sequence (linear 1000 1000))) on (dataset "A" 100 (bits 32) 1000))
   (train (model (sequence (linear 1000 1000))) on (dataset "B" 100 (bits 32) 1000))
   with ((vram 20)))

  (printf "\nTest 4: Auto-Partitioning Strategy\n")
  (pipeline
   (train (model (bits 32) (sequence (linear 1000 1000))) 
          on (dataset "J1" 100 (bits 32) 1000))
   (train (model (bits 32) (sequence (linear 1000 1000))) 
          on (dataset "J2" 100 (bits 32) 1000))
   (train (model (bits 32) (sequence (linear 1000 1000))) 
          on (dataset "J3" 100 (bits 32) 1000))
   with ((vram 40)))

  (printf "\nTest 5: Fine-Grained Component Liveness (Corrected)\n")
  ;; Using 'define-component' automatically assigns the name string.
  (define-component encoder (bits 32) (sequence (linear 500 250 relu)))
  (define-component decoder (bits 32) (sequence (linear 250 500 relu)))
  (define-component generator (bits 32) (sequence (linear 250 250 relu)))

  (pipeline
   (train encoder decoder on (dataset "AutoEnc" 100 (bits 32) 500))
   (train encoder generator on (dataset "GenTrain" 100 (bits 32) 500))
   (train generator decoder on (dataset "Finetune" 100 (bits 32) 250))
   with ((vram 4.5)))

  (printf "\nTest 6: Batch Maximization vs Caching\n")
  (define-component big-model (bits 32) (sequence (linear 1500 1300)))
  (define-component small-model (bits 32) (sequence (linear 100 100)))
  
  (pipeline
   (train big-model on (dataset "Job1" 100 (bits 32) 1500))
   (train small-model on (dataset "Job2" 10000 (bits 32) 100))
   (train big-model on (dataset "Job3" 100 (bits 32) 1500))
   with ((vram 50)))
)