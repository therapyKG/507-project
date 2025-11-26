#lang rosette

(require racket/match)
(require racket/format)
(require "syntax.rkt")

;; ============================================================
;; Memory Estimation Functions
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

;; Calculate memory for ONE sample's activations (for batch size calculation)
(define (estimate-activation-per-sample mdl)
  (estimate-activations-memory mdl 1))

;; Calculate dataset memory if loaded into VRAM
(define (estimate-dataset-memory dset)
  (define dataset-shape (car dset))
  (define dataset-count (cadr dset))
  (define dataset-bits (caddr dset))
  (define bytes-per-value (bits->bytes dataset-bits))
  (* dataset-count (apply * dataset-shape) bytes-per-value))

;; Calculate fixed memory (weights + gradients + optimizer)
(define (estimate-fixed-memory mdl)
  (+ (estimate-weights-memory mdl)
     (estimate-gradients-memory mdl)
     (estimate-optimizer-memory mdl)))

;; Total memory estimate
(define (estimate-total-memory mdl batch-size)
  (+ (estimate-fixed-memory mdl)
     (estimate-activations-memory mdl batch-size)))

;; Total memory including dataset
(define (estimate-total-memory-with-dataset mdl batch-size dset dataset-in-vram-bytes)
  (+ (estimate-fixed-memory mdl)
     (estimate-activations-memory mdl batch-size)
     dataset-in-vram-bytes))

;; Convert bytes to megabytes
(define (bytes->mb bytes)
  (/ bytes (* 1024 1024)))

;; ============================================================
;; Batch Size Optimization
;; ============================================================

;; Calculate maximum possible batch size given VRAM constraint
(define (calculate-max-batch-size mdl dset vram-mb)
  (define vram-bytes (* vram-mb 1024 1024))
  (define fixed-memory (estimate-fixed-memory mdl))
  (define dataset-memory (estimate-dataset-memory dset))
  (define activation-per-sample (estimate-activation-per-sample mdl))
  
  ;; Strategy: Try to fit dataset, then maximize batch size
  ;; If dataset fits with at least batch=1, include it
  ;; Otherwise, maximize batch without dataset
  
  (define memory-with-dataset-and-min-batch
    (+ fixed-memory dataset-memory activation-per-sample))
  
  (if (<= memory-with-dataset-and-min-batch vram-bytes)
      ;; Dataset fits! Calculate max batch with dataset in VRAM
      (let ([remaining (- vram-bytes fixed-memory dataset-memory)])
        (values (quotient remaining activation-per-sample)
                dataset-memory
                #t)) ; dataset fits
      ;; Dataset doesn't fit, calculate max batch without dataset
      (let ([remaining (- vram-bytes fixed-memory)])
        (if (> remaining 0)
            (values (quotient remaining activation-per-sample)
                    0
                    #f) ; dataset doesn't fit
            (values 0 0 #f))))) ; nothing fits!

;; ============================================================
;; Validation Functions
;; ============================================================

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

;; Check if training will fit in VRAM and calculate optimal batch size
(define (validate-memory spec)
  (match spec
    [(train-spec mdl dset config)
     (define vram-mb (cdr (assoc 'vram config)))
     (define requested-batch-size (cdr (assoc 'batch config)))
     
     ;; Calculate what the maximum batch size could be
     (define-values (max-batch dataset-in-vram dataset-fits?)
       (calculate-max-batch-size mdl dset vram-mb))
     
     ;; Calculate memory with requested batch size
     (define total-memory-bytes 
       (estimate-total-memory-with-dataset mdl requested-batch-size dset dataset-in-vram))
     (define total-memory-mb (bytes->mb total-memory-bytes))
     
     (define fixed-memory (estimate-fixed-memory mdl))
     (define dataset-memory (estimate-dataset-memory dset))
     (define activation-memory (estimate-activations-memory mdl requested-batch-size))
     
     (printf "Memory breakdown (MB):\n")
     (printf "  Weights:       ~a\n" (~r (bytes->mb (estimate-weights-memory mdl)) #:precision 2))
     (printf "  Gradients:     ~a\n" (~r (bytes->mb (estimate-gradients-memory mdl)) #:precision 2))
     (printf "  Optimizer:     ~a\n" (~r (bytes->mb (estimate-optimizer-memory mdl)) #:precision 2))
     (printf "  Activations:   ~a (batch=~a)\n" 
             (~r (bytes->mb activation-memory) #:precision 2)
             requested-batch-size)
     (printf "  Dataset:       ~a (~a)\n" 
             (~r (bytes->mb dataset-in-vram) #:precision 2)
             (if dataset-fits? "fits in VRAM" "doesn't fit, streaming from RAM"))
     (printf "  ----------\n")
     (printf "  Total:         ~a MB\n" (~r total-memory-mb #:precision 2))
     (printf "  Available:     ~a MB\n" vram-mb)
     (printf "\n")
     (printf "Batch size analysis:\n")
     (printf "  Requested:     ~a\n" requested-batch-size)
     (printf "  Maximum:       ~a\n" max-batch)
     
     (cond
       [(<= requested-batch-size max-batch)
        (printf "  Status:        ✓ Requested batch size is feasible\n")
        (when (< requested-batch-size max-batch)
          (printf "  Note:          Could use up to ~a for better throughput\n" max-batch))
        #t]
       [(> requested-batch-size max-batch)
        (printf "  Status:        ⚠ OUT OF MEMORY!\n")
        (printf "  Suggestion:    Reduce batch size to ~a or lower\n" max-batch)
        #f]
       [else
        (printf "  Status:        ⚠ Cannot fit even batch size 1!\n")
        #f])]))

;; Validate entire training specification
(define (validate spec)
  (match spec
    [(train-spec mdl dset config)
     (define layers-ok? (validate-layer-dimensions mdl))
     (define dataset-ok? (validate-dataset-match spec))
     (define memory-ok? (validate-memory spec))
     (define param-count (count-model-params mdl))
     (define batch-size (cdr (assoc 'batch config)))
     (define vram-mb (cdr (assoc 'vram config)))
     
     (printf "\n=== Validation Results ===\n")
     (printf "Layers match: ~a\n" layers-ok?)
     (printf "Dataset matches: ~a\n" dataset-ok?)
     (printf "Memory fits: ~a\n" memory-ok?)
     (printf "Total parameters: ~a\n" param-count)
     (printf "Dataset info: ~a datapoints of shape ~a (~a bits)\n"
             (cadr dset) (car dset) (caddr dset))
     (printf "Model bits: ~a\n" (neural-model-bits mdl))
     (printf "Requested batch size: ~a\n" batch-size)
     (printf "VRAM limit: ~a MB\n" vram-mb)
     (and layers-ok? dataset-ok? memory-ok?)]))

;; ============================================================
;; Exports
;; ============================================================

(provide (all-defined-out))