#lang rosette

(require racket/match)

;; ============================================================
;; Core Data Structures
;; ============================================================

(struct layer (type params) #:transparent)
(struct model-layer (layer activation) #:transparent)
(struct neural-model (layers input-shape bits) #:transparent)
(struct train-spec (model dataset config) #:transparent)

;; ============================================================
;; Layer Constructors (non-macro versions for internal use)
;; ============================================================

(define (conv in-channels out-channels kernel-size [stride 1] [padding 0])
  (layer 'conv (list in-channels out-channels kernel-size stride padding)))

(define (maxpool kernel-size [stride #f])
  (layer 'maxpool (list kernel-size (or stride kernel-size))))

(define (dropout rate)
  (layer 'dropout (list rate)))

(define (batchnorm num-features)
  (layer 'batchnorm (list num-features)))

;; ============================================================
;; Activation Functions
;; ============================================================

(define empty 'empty)
(define relu 'relu)
(define tanh 'tanh)
(define sigmoid 'sigmoid)
(define softmax 'softmax)
(define none 'empty)

;; ============================================================
;; Utility Functions
;; ============================================================

;; Convert bits to bytes
(define (bits->bytes bits)
  (/ bits 8))

;; Count parameters in a layer
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
;; DSL Macros - Simplified Syntax
;; ============================================================

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

;; train: top-level construct with optional "with" clause
(define-syntax train
  (syntax-rules (on with vram batch)
    ;; Pattern with vram and batch configuration in nested parens
    [(_ mdl on dset with ((vram v) (batch b)))
     (train-spec mdl dset (list (cons 'vram v) (cons 'batch b)))]
    ;; Pattern without configuration (use defaults)
    [(_ mdl on dset)
     (train-spec mdl dset (list (cons 'vram 8192) (cons 'batch 32)))]))

;; ============================================================
;; Exports
;; ============================================================

(provide (all-defined-out))