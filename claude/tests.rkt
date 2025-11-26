#lang rosette

(require "syntax.rkt")
(require "validate.rkt")

;; ============================================================
;; Test Cases / Examples
;; ============================================================

;; Example 1: Valid autoencoder with 32-bit precision
(define example1
  (train
   (model
    (bits 32)
    (sequence
     (linear 784 128 relu)
     (linear 128 64 tanh)
     (linear 64 128 relu)
     (linear 128 784 sigmoid)))
   on
   (dataset (bits 32) 784 1000)
   with
   ((vram 4096)
    (batch 64))))

(printf "\n=== Example 1: Valid Autoencoder (32-bit, Batch 64) ===\n")
(validate example1)

;; Example 2: Large batch size causing OOM
(define example2
  (train
   (model
    (bits 32)
    (sequence
     (linear 784 2048 relu)
     (linear 2048 2048 relu)
     (linear 2048 784 sigmoid)))
   on
   (dataset (bits 32) 784 5000)
   with
   ((vram 115)    ; Limited VRAM - causes OOM with batch=256
    (batch 256)))) ; Too large for 115 MB

(printf "\n=== Example 2: Out of Memory (Large Batch, Limited VRAM) ===\n")
(validate example2)

;; Example 3: Using 16-bit to save memory
(define example3
  (train
   (model
    (bits 16)
    (sequence
     (linear 784 2048 relu)
     (linear 2048 2048 relu)
     (linear 2048 784 sigmoid)))
   on
   (dataset (bits 16) 784 5000)
   with
   ((vram 512)    ; Same VRAM as example2
    (batch 128)))) ; But using 16-bit

(printf "\n=== Example 3: 16-bit Memory Optimization ===\n")
(validate example3)

;; Example 4: Dimension mismatch
(define example4
  (train
   (model
    (bits 32)
    (sequence
     (linear 784 128 relu)
     (linear 64 10 softmax)))  ; Mismatch: expects 128, gets 64
   on
   (dataset (bits 32) 784 500)
   with
   ((vram 2048)
    (batch 32))))

(printf "\n=== Example 4: Layer Dimension Mismatch ===\n")
(validate example4)

;; Example 5: Ultra-compact 8-bit model
(define example5
  (train
   (model
    (bits 8)
    (sequence
     (linear 784 512 relu)
     (linear 512 256 relu)
     (linear 256 10 softmax)))
   on
   (dataset (bits 8) 784 10000)
   with
   ((vram 256)
    (batch 64))))

(printf "\n=== Example 5: 8-bit Ultra-Compact Model ===\n")
(validate example5)

;; Example 6: Using defaults (no with clause)
(define example6
  (train
   (model
    (sequence
     (linear 100 50 relu)
     (linear 50 10 softmax)))
   on
   (dataset 100 1000)))

(printf "\n=== Example 6: Using Defaults (32-bit, 8192 MB VRAM, Batch 32) ===\n")
(validate example6)

;; Example 7: Small dataset that fits in VRAM with large batch
(define example7
  (train
   (model
    (bits 32)
    (sequence
     (linear 784 256 relu)
     (linear 256 128 relu)
     (linear 128 10 softmax)))
   on
   (dataset (bits 32) 784 1000)  ; Small dataset: ~3MB
   with
   ((vram 1024)   ; 1GB VRAM
    (batch 64))))

(printf "\n=== Example 7: Small Dataset Fits in VRAM ===\n")
(validate example7)

;; Example 8: Large dataset that doesn't fit, shows streaming
(define example8
  (train
   (model
    (bits 32)
    (sequence
     (linear 784 256 relu)
     (linear 256 10 softmax)))
   on
   (dataset (bits 32) 784 100000)  ; Large dataset: ~300MB
   with
   ((vram 128)    ; Only 128MB VRAM
    (batch 32))))

(printf "\n=== Example 8: Large Dataset Doesn't Fit (Streaming) ===\n")
(validate example8)

;; Example 9: Requesting batch size too large - auto-suggest smaller
(define example9
  (train
   (model
    (bits 32)
    (sequence
     (linear 784 2048 relu)
     (linear 2048 2048 relu)
     (linear 2048 10 softmax)))
   on
   (dataset (bits 32) 784 10000)
   with
   ((vram 95)      ; Limited VRAM - causes OOM with batch=512
    (batch 512))))  ; Way too large for 95 MB!

(printf "\n=== Example 9: Batch Size Too Large - Auto Suggestion ===\n")
(validate example9)

;; Example 10: Finding optimal batch size for available VRAM
(define example10
  (train
   (model
    (bits 16)  ; Using 16-bit to be memory efficient
    (sequence
     (linear 784 512 relu)
     (linear 512 256 relu)
     (linear 256 10 softmax)))
   on
   (dataset (bits 16) 784 50000)
   with
   ((vram 512)
    (batch 128))))

(printf "\n=== Example 10: 16-bit Model with Medium Dataset ===\n")
(validate example10)