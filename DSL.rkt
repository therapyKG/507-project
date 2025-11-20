#lang rosette

(require racket/match)
(require racket/format)

;; ============================================================
;; Layer Definitions
;; ============================================================

(struct layer (type params) #:transparent)

(define (linear in-dim out-dim)
  (layer 'linear (list in-dim out-dim)))

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

(struct model-layer (layer activation) #:transparent)
(struct neural-model (layers input-shape) #:transparent)

;; ============================================================
;; DSL Macros
;; ============================================================

(define-syntax model
  (syntax-rules ()
    [(_ (layer-spec ...) on (input-shape ...))
     (neural-model
      (list layer-spec ...)
      (list input-shape ...))]))

;; Macro to create layer-activation pairs
(define-syntax-rule (make-layer-pair layer-expr activation-expr)
  (model-layer layer-expr activation-expr))

;; Allow comma as whitespace
(define-syntax (parse-layers stx)
  (syntax-case stx ()
    [(_ layer act)
     #'(model-layer layer act)]))

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
    [(neural-model layers _)
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
    [(neural-model layers input-shape)
     (define shapes
       (for/fold ([shape input-shape]
                  [shapes (list input-shape)])
                 ([ml layers])
         (define new-shape (compute-output-shape (model-layer-layer ml) shape))
         (values new-shape (append shapes (list new-shape)))))
     shapes]))



 (count-model-params
  (neural-model
   (list (model-layer (linear 784 128) relu)
         (model-layer (linear 128 64) relu)
         (model-layer (linear 64 10) softmax))
   (list 784)))

