#lang rosette

(require racket/match)
(require racket/format)

;; ============================================================
;; Layer Definitions
;; ============================================================

(struct layer (type params) #:transparent)

(define (conv in-channels out-channels kernel-size [stride 1] [padding 0])
  (layer 'conv (list in-channels out-channels kernel-size stride padding)))

  (define (maxpool kernel-size [stride #f])
  (layer 'maxpool (list kernel-size (or stride kernel-size))))

(define (dropout rate)
  (layer 'dropout (list rate)))

(define (batchnorm num-features)
  (layer 'batchnorm (list num-features)))

;; Activation functions
(define empty 'empty)
(define relu 'relu)
(define tanh 'tanh)
(define sigmoid 'sigmoid)
(define softmax 'softmax)

;; Get byte size from bits
(define (bits->bytes bits)
  (/ bits 8))

(struct model-layer (layer activation) #:transparent)
(struct neural-model (layers input-shape bits) #:transparent)

;; ============================================================
;; DSL Macros - Simplified Syntax
;; ============================================================

;; Activation as symbol
(define none 'empty)

;; Combined layer definition: linear(in out activation)
(define-syntax linear
  (syntax-rules (relu tanh sigmoid softmax none)
    [(_ in out relu)
     (model-layer (layer 'linear (list in out)) 'relu)]
    [(_ in out tanh)
     (model-layer (layer 'linear (list in out)) 'tanh)]
    [(_ in out sigmoid)
     (model-layer (layer 'linear (list in out)) 'sigmoid)]
    [(_ in out softmax)
     (model-layer (layer 'linear (list in out)) 'softmax)]
    [(_ in out none)
     (model-layer (layer 'linear (list in out)) 'empty)]))

;; sequence: creates a list of layers
(define-syntax sequence
  (syntax-rules ()
    [(_ layer ...)
     (list layer ...)]))

;; bits specification for model
(define-syntax bits
  (syntax-rules ()
    [(_ n)
     n]))

;; model: wraps layers with bits support
(define-syntax model
  (syntax-rules (sequence bits)
    [(_ (bits n) (sequence layer ...))
     (let ([layers (list layer ...)])
       (define input-dim (car (layer-params (model-layer-layer (car layers)))))
       (neural-model layers (list input-dim) n))]
    [(_ (sequence layer ...))
     (let ([layers (list layer ...)])
       (define input-dim (car (layer-params (model-layer-layer (car layers)))))
       (neural-model layers (list input-dim) 32))]))  ; default to 32 bits

;; dataset: specifies dataset dimensions, count, and bits
(define-syntax dataset
  (syntax-rules (bits)
    [(_ (bits n) dims ... count)
     (list (list dims ...) count n)]
    [(_ dims ... count)
     (list (list dims ...) count 32)]))  ; default to 32 bits

;; training configuration - removed, will be handled directly in train macro

;; train: top-level construct with optional "with" clause
(define-syntax train
  (syntax-rules (on with vram batch)
    ;; Pattern with vram and batch configuration in nested parens
    [(_ mdl on dset with ((vram v) (batch b)))
     (train-spec mdl dset (list (cons 'vram v) (cons 'batch b)))]
    ;; Pattern without configuration (use defaults)
    [(_ mdl on dset)
     (train-spec mdl dset (list (cons 'vram 8192) (cons 'batch 32)))]))

;; Training specification structure
(struct train-spec (model dataset config) #:transparent)

(define (count-layer-params lyr)
  (match lyr
    [(layer 'linear (list in-dim out-dim))
     (+ (* in-dim out-dim) out-dim)] ;; weights + biases
    [(layer 'conv (list in-ch out-ch kernel _ _))
     (+ (* in-ch out-ch kernel kernel) out-ch)] ;; weights + biases
    [(layer 'batchnorm (list num-features))
     (* 2 num-features)] ;; gamma and beta
    [_ 0]))

;; Count total parameters in model
(define (count-model-params mdl)
  (match mdl
    [(neural-model layers _ _)
     (for/sum ([ml layers])
       (count-layer-params (model-layer-layer ml)))]))

;; Get output shape after a layer
(define (compute-output-shape lyr input-shape)
  (match* (lyr input-shape)
    [((layer 'linear (list _ out-dim)) _)
     (list out-dim)]
    [((layer 'conv (list _ out-ch kernel stride padding)) (list in-h in-w))
     (define out-h (quotient (+ in-h (* 2 padding) (- kernel)) stride))
     (define out-w (quotient (+ in-w (* 2 padding) (- kernel)) stride))
     (list out-ch out-h out-w)]
    [((layer 'maxpool (list kernel stride)) (list ch in-h in-w))
     (define out-h (quotient in-h stride))
     (define out-w (quotient in-w stride))
     (list ch out-h out-w)]
    [(_ shape) shape]))

;; Infer shapes through entire model
(define (infer-shapes mdl)
  (match mdl
    [(neural-model layers input-shape _)
     (define-values (final-shape shapes)
       (for/fold ([shape input-shape]
                  [shapes (list input-shape)])
                 ([ml layers])
         (define new-shape (compute-output-shape (model-layer-layer ml) shape))
         (values new-shape (append shapes (list new-shape)))))
     shapes]))

;; ============================================================
;; Validation Functions
;; ============================================================

;; Calculate memory for model weights
(define (estimate-weights-memory mdl)
  (match mdl
    [(neural-model layers _ bits)
     (define param-count (count-model-params mdl))
     (define bytes-per-param (bits->bytes bits))
     (* param-count bytes-per-param)]))

;; Calculate memory for gradients (same size as weights)
(define (estimate-gradients-memory mdl)
  (estimate-weights-memory mdl))

;; Calculate memory for optimizer state (Adam: 2x weights for momentum and variance)
(define (estimate-optimizer-memory mdl)
  (* 2 (estimate-weights-memory mdl)))

;; Calculate memory for activations during forward pass
(define (estimate-activations-memory mdl batch-size)
  (match mdl
    [(neural-model layers input-shape bits)
     (define bytes-per-param (bits->bytes bits))
     (define shapes (infer-shapes mdl))
     ;; Sum up memory for each layer's output
     (for/sum ([shape (cdr shapes)])  ; skip input shape
       (* batch-size (apply * shape) bytes-per-param))]))

;; Total memory estimate
(define (estimate-total-memory mdl batch-size)
  (+ (estimate-weights-memory mdl)
     (estimate-gradients-memory mdl)
     (estimate-optimizer-memory mdl)
     (estimate-activations-memory mdl batch-size)))

;; Convert bytes to megabytes
(define (bytes->mb bytes)
  (/ bytes (* 1024 1024)))

;; Check if all layer dimensions match
(define (validate-layer-dimensions mdl)
  (match mdl
    [(neural-model layers _ _)
     (for/and ([i (in-range (sub1 (length layers)))])
       (define current-layer (list-ref layers i))
       (define next-layer (list-ref layers (add1 i)))
       (define current-out (cadr (layer-params (model-layer-layer current-layer))))
       (define next-in (car (layer-params (model-layer-layer next-layer))))
       (if (= current-out next-in)
           #t
           (begin
             (printf "Dimension mismatch: Layer ~a output (~a) != Layer ~a input (~a)\n"
                     i current-out (add1 i) next-in)
             #f)))]))

;; Check if dataset dimensions match model input
(define (validate-dataset-match spec)
  (match spec
    [(train-spec mdl dset _)
     (define model-input-shape (neural-model-input-shape mdl))
     (define dataset-shape (car dset))
     (if (equal? model-input-shape dataset-shape)
         #t
         (begin
           (printf "Dataset dimension mismatch: Model expects ~a, dataset has ~a\n"
                   model-input-shape dataset-shape)
           #f))]))

;; Check if training will fit in VRAM
(define (validate-memory spec)
  (match spec
    [(train-spec mdl dset config)
     (define vram-mb (cdr (assoc 'vram config)))
     (define batch-size (cdr (assoc 'batch config)))
     (define total-memory-bytes (estimate-total-memory mdl batch-size))
     (define total-memory-mb (bytes->mb total-memory-bytes))
     
     (printf "Memory breakdown (MB):\n")
     (printf "  Weights:       ~a\n" (~r (bytes->mb (estimate-weights-memory mdl)) #:precision 2))
     (printf "  Gradients:     ~a\n" (~r (bytes->mb (estimate-gradients-memory mdl)) #:precision 2))
     (printf "  Optimizer:     ~a\n" (~r (bytes->mb (estimate-optimizer-memory mdl)) #:precision 2))
     (printf "  Activations:   ~a\n" (~r (bytes->mb (estimate-activations-memory mdl batch-size)) #:precision 2))
     (printf "  Total:         ~a MB\n" (~r total-memory-mb #:precision 2))
     (printf "  Available:     ~a MB\n" vram-mb)
     
     (if (<= total-memory-mb vram-mb)
         #t
         (begin
           (printf "⚠ OUT OF MEMORY: Need ~a MB but only ~a MB available!\n"
                   (~r total-memory-mb #:precision 2) vram-mb)
           #f))]))

;; Validate entire training specification
(define (validate spec)
  (match spec
    [(train-spec mdl dset config)
     (define layers-ok? (validate-layer-dimensions mdl))
     (define dataset-ok? (validate-dataset-match spec))
     (define memory-ok? (validate-memory spec))
     (define param-count (count-model-params mdl))
     (define batch-size (cdr (assoc 'batch config)))
     
     (printf "\n=== Validation Results ===\n")
     (printf "Layers match: ~a\n" layers-ok?)
     (printf "Dataset matches: ~a\n" dataset-ok?)
     (printf "Memory fits: ~a\n" memory-ok?)
     (printf "Total parameters: ~a\n" param-count)
     (printf "Dataset info: ~a datapoints of shape ~a (~a bits)\n"
             (cadr dset) (car dset) (caddr dset))
     (printf "Model bits: ~a\n" (neural-model-bits mdl))
     (printf "Batch size: ~a\n" batch-size)
     (and layers-ok? dataset-ok? memory-ok?)]))


;; ============================================================
;; Examples
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
   ((vram 512)    ; Only 512 MB VRAM
    (batch 256)))) ; Large batch

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