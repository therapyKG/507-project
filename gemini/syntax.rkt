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

;; Structs
(struct layer (type params activation) #:transparent)
(struct model-struct (layers bit-width) #:transparent)
(struct dataset-struct (spec count shape bit-width) #:transparent)
(struct train-config (vram-mb batch-size) #:transparent)

;; ============================================================
;; 2. Syntax Helpers (Bits & Parsing)
;; ============================================================

;; The user writes (bits 32), which evaluates to a tagged list.
(define (bits n) (list 'dsl-bits-tag n))

;; Helper to extract bit-width from a list of mixed arguments
(define (extract-bits args [default 32])
  (define found (findf (lambda (x) (and (list? x) 
                                        (not (empty? x)) 
                                        (eq? (car x) 'dsl-bits-tag))) 
                       args))
  (if found (cadr found) default))

;; Helper to remove the bits tag from arguments to get the rest
(define (remove-bits args)
  (filter (lambda (x) (not (and (list? x) 
                                (not (empty? x)) 
                                (eq? (car x) 'dsl-bits-tag)))) 
          args))

;; ============================================================
;; 3. Layer Constructors
;; ============================================================

(define (linear in-dim out-dim [act none])
  (layer 'linear (list in-dim out-dim) act))

(define (conv in-ch out-ch kernel [act none] #:stride [stride 1] #:padding [padding 0])
  (layer 'conv (list in-ch out-ch kernel stride padding) act))

(define (maxpool kernel [stride #f])
  (layer 'maxpool (list kernel (or stride kernel)) none))

(define (flatten)
  (layer 'flatten '() none))

;; ============================================================
;; 4. DSL Structural Elements
;; ============================================================

(define (sequence . layers)
  layers)

;; Model: Accepts (sequence ...) and optional (bits N) in any order
(define (model . args)
  (define bit-width (extract-bits args 32)) ;; Default to 32 bits
  (define clean-args (remove-bits args))
  
  ;; The sequence of layers is expected to be the first remaining argument
  (define layers (if (empty? clean-args) '() (first clean-args)))
  
  (model-struct layers bit-width))

;; Dataset: Accepts spec, count, dims..., and optional (bits N)
(define (dataset spec count . args)
  (define bit-width (extract-bits args 32)) ;; Default to 32 bits
  (define dims (remove-bits args))
  
  (dataset-struct spec count dims bit-width))