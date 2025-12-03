#lang rosette

(require racket/match)
(require racket/format)

;; ============================================================
;; 1. Core Definitions & Activations
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

;; ============================================================
;; 5. Analysis & Validation Logic
;; ============================================================

(define (count-layer-params lyr)
  (match lyr
    [(layer 'linear (list in out) _)
     (+ (* in out) out)]
    [(layer 'conv (list in out k s p) _)
     (+ (* in out k k) out)]
    [_ 0]))

(define (validate-layer lyr input-shape)
  (match* (lyr input-shape)
    [((layer 'linear (list in out) _) (list curr-dim))
     (if (= in curr-dim)
         (values (list out) #f)
         (values (list out) (format "Dimension Mismatch: Linear layer expects ~a, got ~a" in curr-dim)))]
    [((layer 'linear (list in out) _) other)
     (values (list out) (format "Shape Mismatch: Linear expects 1D, got ~a" other))]
    
    [((layer 'conv (list in out k s p) _) (list c h w))
     (define next-h (quotient (+ h (* 2 p) (- k)) s))
     (define next-w (quotient (+ w (* 2 p) (- k)) s))
     (define next-shape (list out next-h next-w))
     (if (= in c)
         (values next-shape #f)
         (values next-shape (format "Channel Mismatch: Conv expects ~a ch, got ~a" in c)))]
    [((layer 'conv (list in out k s p) _) other)
     (values (list out 1 1) (format "Shape Mismatch: Conv expects 3D (C,H,W), got ~a" other))]
     
    [((layer 'maxpool (list k s) _) (list c h w))
     (values (list c (quotient h s) (quotient w s)) #f)]
    [((layer 'maxpool _ _) other)
     (values (list 1 1 1) (format "Shape Mismatch: Maxpool expects 3D, got ~a" other))]

    [((layer 'flatten _ _) shape)
     (if (list? shape) (values (list (apply * shape)) #f) (values '(1) "Invalid flatten input"))]
     
    [(_ _) (values '(0) "Unknown layer")]))

(define (analyze-training mdl data [config #f])
  (define input-shape (dataset-struct-shape data))
  (define layers (model-struct-layers mdl))
  
  ;; Get bit widths directly
  (define data-bits (dataset-struct-bit-width data))
  (define model-bits (model-struct-bit-width mdl))
  (define model-bytes (/ model-bits 8.0))
  
  ;; Config defaults
  (define vram-limit-mb (if config (train-config-vram-mb config) 100000))
  (define batch-size (if config (train-config-batch-size config) 1))
  
  (printf "\n=== Training Analysis ===\n")
  (printf "Dataset: ~a | Count: ~a | Input: ~a | Precision: ~a-bit\n" 
          (dataset-struct-spec data) (dataset-struct-count data) input-shape data-bits)
  (printf "Model Precision: ~a-bit (~a bytes/param)\n" model-bits model-bytes)
  
  (when config
    (printf "Config: Batch Size ~a | VRAM Limit ~a MB\n" batch-size vram-limit-mb))

  (define total-params 0)
  
  ;; CORRECTION 1: Initialize activation sum with the input data size.
  ;; We need to store the input batch to compute gradients for the first layer.
  (define input-elements (if (empty? input-shape) 0 (apply * input-shape)))
  (define total-activation-elements input-elements)
  
  ;; 1. Architecture Validation Loop
  (define-values (final-shape error-count)
    (for/fold ([current-shape input-shape] [err-count 0])
              ([l layers] [i (in-naturals 1)])
      
      (set! total-params (+ total-params (count-layer-params l)))
      (define-values (next-shape error-msg) (validate-layer l current-shape))
      
      (define layer-output-elements (apply * next-shape))
      
      ;; CORRECTION 2: Do not count memory for zero-copy layers like flatten.
      (unless (eq? (layer-type l) 'flatten)
        (set! total-activation-elements (+ total-activation-elements layer-output-elements)))
      
      (printf "Layer ~a (~a): ~a -> ~a" i (layer-type l) current-shape next-shape)
      (when error-msg (printf " [ERROR: ~a]" error-msg))
      (newline)
      
      (values next-shape (if error-msg (+ err-count 1) err-count))))

  ;; 2. Memory Estimation
  (define static-mem-bytes (* total-params model-bytes 4)) ;; Optimizers often need copies
  (define dynamic-mem-bytes (* total-activation-elements batch-size model-bytes 2)) ;; Fwd + Bwd
  
  (define total-mem-bytes (+ static-mem-bytes dynamic-mem-bytes))
  (define total-mem-mb (/ total-mem-bytes 1024.0 1024.0))
  
  (printf "-------------------------\n")
  (printf "Memory Estimation:\n")
  (printf "  Params: ~a (~a MB static)\n" 
          total-params (~r (/ static-mem-bytes 1024.0 1024.0) #:precision 2))
  (printf "  Activations: ~a elements/batch (includes inputs)\n" total-activation-elements)
  (printf "  Total VRAM Required: ~a MB\n" (~r total-mem-mb #:precision 2))
  
  (define mem-ok? (<= total-mem-mb vram-limit-mb))
  
  (printf "=========================\n")
  (cond
    [(> error-count 0)
     (printf "Status: FAILED (Architecture Errors)\n") #f]
    [(not mem-ok?)
     (printf "Status: FAILED (OOM - Requires ~a MB, Limit ~a MB)\n" 
             (~r total-mem-mb #:precision 1) vram-limit-mb) #f]
    [else
     (printf "Status: SUCCESS\n") #t]))

;; ============================================================
;; 6. Top-Level Macro
;; ============================================================

(define-syntax train
  (syntax-rules (on with vram batch)
    [(_ model-expr on dataset-expr with ((vram v-val) (batch b-val)))
     (analyze-training model-expr dataset-expr (train-config v-val b-val))]
    [(_ model-expr on dataset-expr)
     (analyze-training model-expr dataset-expr #f)]))

;; ============================================================
;; 7. Examples
;; ============================================================

(module+ main
  (printf "Test 1: Standard 32-bit Training\n")
  (train
   (model (bits 32)
    (sequence
     (linear 784 128 relu)
     (linear 128 10 softmax)))
   on
   (dataset "MNIST" 60000 (bits 32) 784)
   with ((vram 96) (batch 128)))

  (printf "\nTest 2: High Res 16-bit (Low Memory Check)\n")
  (train
   (model 
    (bits 16)
    (sequence
     (conv 3 64 3 relu)
     (conv 64 64 3 relu)
     (maxpool 2)
     (flatten)
     (linear 16777216 10)))
   on
   (dataset "HD-Image" 1000 3 1024 1024 (bits 16)) ;; (bits) can be at end
   with ((vram 2000) (batch 8))))