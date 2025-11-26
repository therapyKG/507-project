#lang rosette

(require "syntax.rkt")
(require "validate.rkt")

;; ============================================================
;; Simple Example: Understanding Memory and Batch Optimization
;; ============================================================

(printf "========================================\n")
(printf "Neural Network DSL - Feature Demo\n")
(printf "========================================\n")

;; ============================================================
;; Demo 1: Small Dataset - Everything Fits
;; ============================================================

(printf "\n\n>>> Demo 1: Small Dataset (3 MB) with 1 GB VRAM <<<\n")
(printf "Expected: Dataset fits, large max batch size\n\n")

(define demo1
  (train
   (model
    (bits 32)
    (sequence
     (linear 784 256 relu)
     (linear 256 10 softmax)))
   on
   (dataset (bits 32) 784 1000)  ; ~3 MB dataset
   with
   ((vram 1024)   ; 1 GB VRAM
    (batch 64))))

(validate demo1)

;; ============================================================
;; Demo 2: Large Dataset - Must Stream
;; ============================================================

(printf "\n\n>>> Demo 2: Large Dataset (300 MB) with 128 MB VRAM <<<\n")
(printf "Expected: Dataset doesn't fit, will stream, max batch still large\n\n")

(define demo2
  (train
   (model
    (bits 32)
    (sequence
     (linear 784 256 relu)
     (linear 256 10 softmax)))
   on
   (dataset (bits 32) 784 100000)  ; ~300 MB dataset
   with
   ((vram 128)
    (batch 32))))

(validate demo2)

;; ============================================================
;; Demo 3: Batch Size Too Large
;; ============================================================

(printf "\n\n>>> Demo 3: Requesting Batch Size That's Too Large <<<\n")
(printf "Expected: Validation fails, suggests smaller batch\n\n")

(define demo3
  (train
   (model
    (bits 32)
    (sequence
     (linear 784 1024 relu)
     (linear 1024 1024 relu)
     (linear 1024 10 softmax)))
   on
   (dataset (bits 32) 784 10000)
   with
   ((vram 32)      ; Only 32 MB - now it will definitely fail!
    (batch 512))))  ; Too large for 32 MB

(validate demo3)

;; ============================================================
;; Demo 4: Using 16-bit to Fit More
;; ============================================================

(printf "\n\n>>> Demo 4: Same Model as Demo 3, but with 16-bit <<<\n")
(printf "Expected: 16-bit uses half the memory, larger batch possible\n\n")

(define demo4
  (train
   (model
    (bits 16)  ; Half the memory!
    (sequence
     (linear 784 1024 relu)
     (linear 1024 1024 relu)
     (linear 1024 10 softmax)))
   on
   (dataset (bits 16) 784 10000)
   with
   ((vram 32)      ; Same 32 MB as Demo 3
    (batch 512))))  ; Now this should fit with 16-bit!

(validate demo4)

;; ============================================================
;; Summary
;; ============================================================

(printf "\n\n========================================\n")
(printf "Key Takeaways:\n")
(printf "========================================\n")
(printf "1. Dataset memory is considered but optional\n")
(printf "   - If it fits: loaded to VRAM for speed\n")
(printf "   - If not: streamed from RAM (slightly slower)\n\n")
(printf "2. Max batch size is automatically calculated\n")
(printf "   - Based on: model size + activations + dataset\n")
(printf "   - Validator tells you if batch is too large\n\n")
(printf "3. Precision matters for memory\n")
(printf "   - 32-bit: 4 bytes per value\n")
(printf "   - 16-bit: 2 bytes per value (50%% savings)\n")
(printf "   - 8-bit: 1 byte per value (75%% savings)\n\n")
(printf "4. Activation memory scales with batch size\n")
(printf "   - Double batch → double activation memory\n")
(printf "   - Model memory stays constant\n\n")
(printf "5. Demo 3 vs Demo 4 comparison:\n")
(printf "   - Same model, same VRAM (32 MB)\n")
(printf "   - Demo 3 (32-bit): batch=512 fails\n")
(printf "   - Demo 4 (16-bit): batch=512 succeeds!\n")
(printf "   - 16-bit allows 2x larger batches\n")
(printf "========================================\n")