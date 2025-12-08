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

;; Keywords
;; We define them as syntax that raises errors to prevent accidental evaluation
(define-syntax (with stx) (raise-syntax-error #f "Keyword 'with' used as an expression." stx))
(define-syntax (vram stx) (raise-syntax-error #f "Keyword 'vram' used as an expression." stx))
(define-syntax (strategy stx) (raise-syntax-error #f "Keyword 'strategy' used as an expression." stx))
(define on 'on)

;; Structs
(struct layer (type params activation) #:transparent)
(struct model-struct (layers bit-width name uid) #:transparent)
(struct dataset-struct (spec count shape bit-width) #:transparent)
(struct train-job (models dataset) #:transparent) 

;; Updated Config Struct to support multiple fields
(struct pipeline-config (vram-mb strategy) #:transparent)

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

(define (model . args)
  (define-values (name remaining-args)
    (if (and (not (empty? args)) (string? (first args)))
        (values (first args) (rest args))
        (values "Anonymous" args)))

  (define bit-width (extract-bits remaining-args 32))
  (define layers (remove-bits remaining-args))
  
  (model-struct layers bit-width name (gensym 'model)))

(define-syntax define-component
  (syntax-rules ()
    [(_ name args ...)
     (define name (model (symbol->string 'name) args ...))]))

(define (dataset spec count . args)
  (define bit-width (extract-bits args 32))
  (define dims (remove-bits args))
  (dataset-struct spec count dims bit-width))

;; ============================================================
;; 4. Train Macro
;; ============================================================

(define (parse-train-args args)
  (define on-clause-idx (index-where args (lambda (x) (eq? x 'on))))
  (unless on-clause-idx (error "Missing 'on' keyword in train command"))
  
  (define models (take args on-clause-idx))
  (define dataset (list-ref args (+ on-clause-idx 1)))
  (train-job models dataset))

(define-syntax train
  (syntax-rules (on)
    [(_ item ...)
     (parse-train-args (list item ...))]))