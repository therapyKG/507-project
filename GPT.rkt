#lang rosette
;; Rosette DSL for small neural nets + training-time memory optimization
;; by symbolic checkpoint decisions (keep-params & keep-activations).
;; Assumes single minibatch training: forward then backward.

(require racket/match)
(require racket/format)
(require racket/list)
;; ============================================================
;; Layer & Model definitions (transparent structs)
;; ============================================================
(struct layer (type params) #:transparent)
(struct model-layer (layer activation) #:transparent)
(struct neural-model (layers input-shape) #:transparent)

;; Helper constructors
(define (conv in-ch out-ch kernel-size #:stride [stride 1] #:padding [padding 0])
  (layer 'conv (list in-ch out-ch kernel-size stride padding)))

(define (maxpool kernel-size #:stride [stride #f])
  (layer 'maxpool (list kernel-size (or stride kernel-size))))

(define (linear in-dim out-dim)
  (layer 'linear (list in-dim out-dim)))

(define (dropout rate)
  (layer 'dropout (list rate)))

(define (batchnorm num-feats)
  (layer 'batchnorm (list num-feats)))

;; Activation tags (symbols)
(define empty 'empty)
(define relu 'relu)
(define tanh 'tanh)
(define sigmoid 'sigmoid)
(define softmax 'softmax)

;; ============================================================
;; Size / shape computations (concrete arithmetic)
;; ============================================================
;; Parameter count per layer (integer)
(define (params-count lyr)
  (match lyr
    [(layer 'linear (list in-dim out-dim))
     (+ (* in-dim out-dim) out-dim)] ;; W + b
    [(layer 'conv (list in-ch out-ch kernel stride padding))
     (+ (* in-ch out-ch kernel kernel) out-ch)]
    [(layer 'batchnorm (list num-feats))
     (* 2 num-feats)]
    [_ 0]))

;; Activation output shape given input shape
;; input-shape: for linear: (list n) ; for conv: (list channels height width)
(define (compute-output-shape lyr input-shape)
  (match* (lyr input-shape)
    [((layer 'linear (list _ out-dim)) (list _))
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

;; Activation size in memory (assume float32 -> 4 bytes per element)
(define FLOAT-BYTES 4)
(define (activation-size-bytes shape batch-size)
  (match shape
    [(list n) (* n batch-size FLOAT-BYTES)]
    [(list ch h w) (* ch h w batch-size FLOAT-BYTES)]
    [_ 0]))

;; Count total parameters bytes (assuming float32)
(define (params-bytes-count lyr)
  (* (params-count lyr) FLOAT-BYTES))

;; ============================================================
;; Helper: infer shapes through model (purely concrete)
;; ============================================================
(define (infer-shapes mdl)
  (match mdl
    [(neural-model layers input-shape)
     (let loop ([rest layers] [shape input-shape] [acc (list input-shape)])
       (if (null? rest)
           acc
           (let* ([ml (car rest)]
                  [lyr (model-layer-layer ml)]
                  [new-shape (compute-output-shape lyr shape)])
             (loop (cdr rest) new-shape (append acc (list new-shape))))))]))

;; ============================================================
;; Macro to create top-level vectors of symbolic booleans
;; Must expand at top-level because Rosette requires symbolic defs at module level.
;; Usage:
;;   (define-symbolic-bools keep-param-vec 3) ; defines keep-param-vec as a vector of 3 symbolic booleans
;; ============================================================
(define-syntax-rule (define-symbolic-bools name n)
  (define name
    (for/vector ([i (in-range n)])
      (let ()
        (define-symbolic* v boolean?)
        v))))

;; ============================================================
;; Memory & constraints builder
;; NOTE: keep-param-vec and keep-act-vec must be top-level symbolic vectors
;; (create them with define-symbolic-bools before calling optimize-checkpoints).
;; ============================================================
;; ============================================================
;; Clean, correct optimize-checkpoints for Rosette
;; ============================================================


;; Clean, correct optimize-checkpoints for Rosette
(define (optimize-checkpoints model batch-size keep-param-vec keep-act-vec
                              #:max-recompute-dist [max-recompute-dist 2])
  (match model
    [(neural-model layers input-shape)
     (define N (length layers))

     ;; 1. Sanity checks
     (unless (= (vector-length keep-param-vec) N)
       (error "keep-param-vec must have length N"))
     (unless (= (vector-length keep-act-vec) (add1 N))
       (error "keep-act-vec must have length N+1 (input activation included)"))

     ;; 2. Compute shapes and byte sizes (Concrete Arithmetic)
     (define shapes (infer-shapes model))
     (define act-bytes
       (for/list ([i (in-range (add1 N))])
         (activation-size-bytes (list-ref shapes i) batch-size)))

     (define param-bytes
       (for/list ([i (in-range N)])
         (params-bytes-count (model-layer-layer (list-ref layers i)))))

     ;; 3. Input activation must be kept (Constraint)
     (assert (vector-ref keep-act-vec 0))

     ;; 4. Activation Availability Constraints
     (for ([i (in-range 1 (add1 N))])
       (define possible-checks
         (for/list ([j (in-range 0 i)])
           (if (<= (- i j) max-recompute-dist)
               (vector-ref keep-act-vec j)
               #f)))

       ;; symbolic OR chain
       (define recomputable?
         (foldr (λ (x acc) (|| x acc)) #f possible-checks))

       ;; enforce: keep-act[i] OR recomputable?
       (assert (|| (vector-ref keep-act-vec i)
                   recomputable?)))

     ;; 5. Peak Memory Expression (Symbolic Calculation)

     ;; Calculate the symbolic terms for parameter memory
     (define param-terms
       (for/list ([i (in-range N)])
         ;; Use Rosette 'if' for symbolic conditional addition
         (if (vector-ref keep-param-vec i)
             (list-ref param-bytes i)
             0)))
    ;; 5. Peak Memory Expression (Symbolic Calculation)

;; Start with 0. Accumulate parameter bytes first.
(define total-param-bytes
  (for/fold ([current-mem 0])
            ([i (in-range N)])
    ;; Rosette 'if' creates the symbolic addition only if the param is kept
    (if (vector-ref keep-param-vec i)
        (+ current-mem (list-ref param-bytes i))
        current-mem)))

;; Accumulate activation bytes, starting from the total parameter memory.
(define peak-memory
  (for/fold ([current-mem total-param-bytes])
            ([i (in-range (add1 N))])
    ;; Rosette 'if' creates the symbolic addition only if the activation is kept
    (if (vector-ref keep-act-vec i)
        (+ current-mem (list-ref act-bytes i))
        current-mem)))

;; Note: You don't need a separate 'total-act-bytes' variable with this method
;; but you can define it if needed for the return value.
(define total-act-bytes
  (for/list ([i (in-range (add1 N))])
    (if (vector-ref keep-act-vec i) (list-ref act-bytes i) 0)))
(define total-act-bytes-sum (foldr + 0 total-act-bytes)) 

;; Now use the final accumulated term for minimization:
;; ----- Optimize -----
(define sol
  (optimize
    #:minimize peak-memory ; Use the accumulated term
    #:guarantee #t))
     ;; 7. Return everything
     (values sol
             keep-param-vec
             keep-act-vec
             param-bytes
             act-bytes
             peak-memory)]))
;; ============================================================
;; Example usage
;; ============================================================
;; Build a small example model similar to your original:
(define ex-model
  (neural-model
   (list (model-layer (linear 784 128) relu)
         (model-layer (linear 128 64) relu)
         (model-layer (linear 64 10) softmax))
   (list 784))) ;; input shape is vector length 784

;; Determine N and define symbolic vectors at top-level
(define N (length (neural-model-layers ex-model)))

;; create top-level symbolic vectors named keep-p and keep-a
(define-symbolic-bools keep-p N)       ; keep-p is a vector length N (params for each layer)
(define-symbolic-bools keep-a (add1 N)) ; keep-a is vector length N+1 (activations: input + each layer output)

;; Run optimizer for a batch-size of 32 and max recompute distance 2
(define-values (sol keep-p-v keep-a-v param-bytes act-bytes peak) (optimize-checkpoints ex-model 32 keep-p keep-a #:max-recompute-dist 2))

;; Print result summary
(printf "Solver object: ~s\n" sol)

(for ([i (in-range (vector-length keep-p-v))])
  (printf "layer ~a: params-bytes=~a keep-param=~a\n"
          i
          (list-ref param-bytes i)
          (evaluate (vector-ref keep-p-v i) sol))) ;; CORRECTED: use evaluate

(for ([i (in-range (vector-length keep-a-v))])
  (printf "act ~a: act-bytes=~a keep-act=~a\n"
          i
          (list-ref act-bytes i)
          (evaluate (vector-ref keep-a-v i) sol))) ;; CORRECTED: use evaluate

(printf "Peak memory (bytes) symbolic expr: ~s\n" peak)
(printf "Model param bytes per layer: ~s\n" param-bytes)
(printf "Activation bytes per shape index: ~s\n" act-bytes)