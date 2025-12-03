#lang rosette

(provide (all-defined-out))

;; ============================================================
;; 1. Core Definitions & Constants
;; ============================================================

(define relu 'relu)
(define tanh 'tanh)
(define sigmoid 'sigmoid)
(define softmax 'softmax)
(define none 'none)

;; Keywords: Define as syntax that errors if evaluated directly.
;; This prevents "unbound identifier" errors while ensuring they act as keywords.
(define-syntax (with stx) (raise-syntax-error #f "Keyword 'with' used as an expression. It must be used inside (pipeline ...)." stx))
(define-syntax (vram stx) (raise-syntax-error #f "Keyword 'vram' used as an expression." stx))
(define on 'on)

;; Structs
(struct layer (type params activation) #:transparent)
(struct model-struct (layers bit-width) #:transparent)
(struct dataset-struct (spec count shape bit-width) #:transparent)

;; New Structures for Pipeline
(struct train-job (model dataset) #:transparent)
(struct pipeline-config (vram-mb) #:transparent)

;; ============================================================
;; 2. Syntax Helpers
;; ============================================================

(define (bits n) (list 'dsl-bits-tag n))

(define (extract-bits args [default 32])
  (define found (findf (lambda (x) (and (list? x) 
                                        (not (empty? x)) 
                                        (eq? (car x) 'dsl-bits-tag))) 
                       args))
  (if found (cadr found) default))

(define (remove-bits args)
  (filter (lambda (x) (not (and (list? x) 
                                (not (empty? x)) 
                                (eq? (car x) 'dsl-bits-tag)))) 
          args))

;; ============================================================
;; 3. Constructors
;; ============================================================

(define (linear in-dim out-dim [act none])
  (layer 'linear (list in-dim out-dim) act))

(define (conv in-ch out-ch kernel [act none] #:stride [stride 1] #:padding [padding 0])
  (layer 'conv (list in-ch out-ch kernel stride padding) act))

(define (maxpool kernel [stride #f])
  (layer 'maxpool (list kernel (or stride kernel)) none))

(define (flatten)
  (layer 'flatten '() none))

(define (sequence . layers)
  layers)

(define (model . args)
  (define bit-width (extract-bits args 32))
  (define clean-args (remove-bits args))
  (define layers (if (empty? clean-args) '() (first clean-args)))
  (model-struct layers bit-width))

(define (dataset spec count . args)
  (define bit-width (extract-bits args 32))
  (define dims (remove-bits args))
  (dataset-struct spec count dims bit-width))

;; ============================================================
;; 4. Train Macro (Data Construction Only)
;; ============================================================

(define-syntax train
  (syntax-rules (on)
    [(_ model-expr on dataset-expr)
     (train-job model-expr dataset-expr)]))