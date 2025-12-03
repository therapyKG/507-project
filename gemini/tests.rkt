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
   ;; Conv output: (32-3+1)=30 -> Pool(2): 15 -> 15*15*32 = 7200
   (train (model (bits 16) (sequence (conv 3 32 3 relu) (maxpool 2) (flatten) (linear 7200 10)))
          on (dataset "CIFAR" 2000 (bits 16) 3 32 32))
   (train (model (bits 32) (sequence (linear 1024 1024 relu) (linear 1024 10)))
          on (dataset "VectorData" 2000 (bits 32) 1024))
   with ((vram 50))) 
  
  (printf "\nTest 3: Impossible Pipeline (OOM)\n")
  ;; This SHOULD fail with CRITICAL ERROR as part of the test
  (pipeline
   (train (model (sequence (linear 1000 1000))) on (dataset "A" 100 (bits 32) 1000))
   (train (model (sequence (linear 1000 1000))) on (dataset "B" 100 (bits 32) 1000))
   with ((vram 20)))

  (printf "\nTest 4: Auto-Partitioning Strategy\n")
  ;; Scenario: 3 Jobs. 
  ;; Each job: 1M params -> ~4MB weights -> ~16MB training memory.
  ;; Total: 48MB static memory.
  ;; VRAM: 50MB.
  ;; Since we need batch size >= 1, we can likely fit 2 jobs (32MB + overhead), but not 3 (48MB + overhead > 50MB).
  (pipeline
   ;; Job 1
   (train (model (bits 32) (sequence (linear 1000 1000))) 
          on (dataset "J1" 100 (bits 32) 1000))
   ;; Job 2
   (train (model (bits 32) (sequence (linear 1000 1000))) 
          on (dataset "J2" 100 (bits 32) 1000))
   ;; Job 3
   (train (model (bits 32) (sequence (linear 1000 1000))) 
          on (dataset "J3" 100 (bits 32) 1000))
   with ((vram 40)))
)