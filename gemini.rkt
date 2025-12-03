#lang rosette
(require racket/match)
(require racket/format)
(require rosette) ; Required for all solver functions

;; ============================================================
;; Structs and Definitions (Unchanged)
;; ============================================================

(struct layer (type params) #:transparent)
(define (linear in-dim out-dim) (layer 'linear (list in-dim out-dim)))
(define (conv in-channels out-channels kernel-size [stride 1] [padding 0]) (layer 'conv (list in-channels out-channels kernel-size stride padding)))
(define (maxpool kernel-size [stride #f]) (layer 'maxpool (list kernel-size (or stride kernel-size))))
(define (dropout rate) (layer 'dropout (list rate)))
(define (batchnorm num-features) (layer 'batchnorm (list num-features)))

(define empty 'empty)
(define relu 'relu)
(define tanh 'tanh)
(define sigmoid 'sigmoid)
(define softmax 'softmax)

(struct model-layer (layer activation) #:transparent)
(struct neural-model (layers input-shape) #:transparent)

(define-syntax model
  (syntax-rules ()
    [(_ layer-spec ... on (input-shape ...))
      (neural-model
      (list layer-spec ...)
      (list input-shape ...))]))

;; ============================================================
;; Helper Functions (Unchanged)
;; ============================================================

(define (count-layer-params lyr)
  (define bytes-per-param 4)
  (match lyr
    [(layer 'linear (list in-dim out-dim)) (* bytes-per-param (+ (* in-dim out-dim) out-dim))]
    [(layer 'conv (list in-ch out-ch kernel _ _)) (* bytes-per-param (+ (* in-ch out-ch kernel kernel) out-ch))]
    [(layer 'batchnorm (list num-features)) (* bytes-per-param (* 2 num-features))]
    [_ 0]))

(define (count-model-params mdl)
  (match mdl
    [(neural-model layers _) (for/sum ([ml layers]) (count-layer-params (model-layer-layer ml)))]))

(define (compute-output-shape lyr input-shape)
  (match* (lyr input-shape)
    [((layer 'linear (list _ out-dim)) _) (list out-dim)]
    [(_ shape) shape]))

(define (infer-shapes mdl)
  (match mdl
    [(neural-model layers input-shape)
      (define-values (shape shapes)
        (for/fold ([shape input-shape] [shapes (list input-shape)])
                  ([ml layers])
          (define new-shape (compute-output-shape (model-layer-layer ml) shape))
          (values new-shape (append shapes (list new-shape)))))
      shapes]))

(define (tensor-size shape)
  (define batch-size 32)
  (define bytes-per-element 4)
  (* bytes-per-element batch-size (apply * shape)))

(define (get-activation-sizes mdl)
  (define shapes (infer-shapes mdl))
  (map tensor-size (cdr shapes)))

(define (compute-total-memory mdl live-activations activation-sizes param-sizes)
  (define total-param-mem (apply + param-sizes))
  (define symbolic-terms
    (for/list ([size activation-sizes] [is-live (vector->list live-activations)])
      (* size is-live)))
  (define live-activation-mem (apply + symbolic-terms))
  (+ total-param-mem live-activation-mem))

;; ============================================================
;; SAT Query (Unchanged)
;; ============================================================

(define (build-sat-query mdl total-memory-limit target-kept)
  (define layers (neural-model-layers mdl))
  (define n (length layers))

  (define activation-sizes (get-activation-sizes mdl))
  (define param-sizes
    (map (lambda (ml) (count-layer-params (model-layer-layer ml))) layers))

  (define (get-symbolic-int) (define-symbolic* sym-var integer?) sym-var)
  (define live-activations (build-vector n (lambda (i) (get-symbolic-int))))

  (begin
    (for ([is-live (vector->list live-activations)])
      (assert (or (= is-live 0) (= is-live 1))))

    (define peak-memory (compute-total-memory mdl live-activations activation-sizes param-sizes))
    (assert (<= peak-memory total-memory-limit))

    (assert (= (vector-ref live-activations (- n 1)) 1))
    
    (define total-kept-activations (apply + (vector->list live-activations)))
    (assert (= total-kept-activations target-kept))
    
    (list live-activations)
    )
  )

;; ============================================================
;; Iterative Search (Final Corrected Logic)
;; ============================================================

(define (find-max-kept mdl total-memory-limit)
  (define n (length (neural-model-layers mdl)))
  
  (define (search-down k)
    (if (< k 1)
        #f
        (begin
          (displayln (format "  Trying to keep ~a activations..." k))
          (define query (build-sat-query mdl total-memory-limit k))
          (define solution (solve query))
          
          (if (sat? solution)
              ;; SUCCESS BRANCH: Using the standard Rosette accessor: model-value
              (let* ([raw-activations (**model-value** solution 'live-activations)] 
                     [kept-activations (vector->list raw-activations)])
                (displayln (format "  Found solution keeping ~a activations." k))
                (list k kept-activations))
              
              ;; FAILURE BRANCH: Recurse down to k-1
              (search-down (- k 1))))))

  (displayln (format "Starting iterative search (recursively) from ~a activations..." n))
  (search-down n))


;; ============================================================
;; Example Usage (Unchanged)
;; ============================================================

(define mlp-model
  (model
   (model-layer (linear 784 128) relu)
   (model-layer (linear 128 64) relu)
   (model-layer (linear 64 10) softmax)
   on (784)))

(displayln "=== Model Analysis (MLP) ===")
(displayln (format "Total Layers (Activations to manage): ~a" (length (neural-model-layers mlp-model))))
(displayln (format "Total Parameters Memory: ~a Bytes" (count-model-params mlp-model)))
(displayln (format "Activation Sizes (L1 to L3): ~a Bytes" (get-activation-sizes mlp-model)))
(displayln "---")

(define MAX-MEMORY-LIMIT 500000) ; 500 KB limit

(displayln (format "Max Memory Limit: ~a Bytes (~a KB)" MAX-MEMORY-LIMIT (quotient MAX-MEMORY-LIMIT 1024)))

(define result (find-max-kept mlp-model MAX-MEMORY-LIMIT))

(match result
  [(list num-kept kept-activations)
   (displayln "\n✅ Optimal Memory Configuration Found (via Recursive SAT Search)!")
   (displayln (format "  Optimal Kept Activations: ~a" num-kept))
   (displayln (format "  Decision Vector (1=Keep, 0=Discard): ~a" kept-activations))]
  [#f (displayln "\n❌ No valid configuration could be found.")]
  )