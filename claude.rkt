#lang rosette

(require racket/match)
(require racket/format)
(require racket/set)

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
;; Memory Management Structures
;; ============================================================

;; Represents a tensor in memory during training
(struct tensor-ref (id layer-idx type size) #:transparent)
;; type: 'input, 'activation, 'gradient, 'weight, 'bias

;; Training phase dependency graph
(struct dependency-graph (forward-deps backward-deps) #:transparent)

;; Memory state at a point in time
(struct memory-state (tensors total-size) #:transparent)

;; ============================================================
;; DSL Macros
;; ============================================================

(define-syntax model
  (syntax-rules ()
    [(_ (layer-spec ...) on (input-shape ...))
     (neural-model
      (list layer-spec ...)
      (list input-shape ...))]))

(define-syntax-rule (make-layer-pair layer-expr activation-expr)
  (model-layer layer-expr activation-expr))

;; ============================================================
;; Parameter Counting
;; ============================================================

(define (count-layer-params lyr)
  (match lyr
    [(layer 'linear (list in-dim out-dim))
     (+ (* in-dim out-dim) out-dim)]
    [(layer 'conv (list in-ch out-ch kernel _ _))
     (+ (* in-ch out-ch kernel kernel) out-ch)]
    [(layer 'batchnorm (list num-features))
     (* 2 num-features)]
    [_ 0]))

(define (count-model-params mdl)
  (match mdl
    [(neural-model layers _)
     (for/sum ([ml layers])
       (count-layer-params (model-layer-layer ml)))]))

;; ============================================================
;; Shape Inference
;; ============================================================

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

(define (infer-shapes mdl)
  (match mdl
    [(neural-model layers input-shape)
     (for/fold ([shapes (list input-shape)])
               ([ml layers])
       (define last-shape (last shapes))
       (define new-shape (compute-output-shape (model-layer-layer ml) last-shape))
       (append shapes (list new-shape)))]))

;; ============================================================
;; Memory Size Calculation
;; ============================================================

(define (shape->size shape)
  (apply * shape))

;; ============================================================
;; Dependency Analysis
;; ============================================================

;; Build forward pass dependencies
(define (build-forward-dependencies mdl)
  (match mdl
    [(neural-model layers input-shape)
     (define shapes (infer-shapes mdl))
     (define n (length layers))
     
     (for/hash ([i (in-range (+ n 1))])
       (values i
               (cond
                 [(= i 0) (set)] ;; Input has no dependencies
                 [else (set (- i 1))])))]))  ;; Each layer depends on previous

;; Build backward pass dependencies (what needs to be kept for backprop)
;; Build backward pass dependencies (what needs to be kept for backprop)
(define (build-backward-dependencies mdl)
  (match mdl
    [(neural-model layers input-shape)
     (define shapes (infer-shapes mdl))
     (define n (length layers))
     
     ;; For backward pass: layer i needs:
     ;; - Gradient from layer i+1 (or 'loss for the last layer)
     ;; - Activation from layer i (for computing gradient)
     ;; - Input to layer i (from layer i-1; for i=0 use input index 0)
     (for/hash ([i (in-range n)])
       (define next (if (< i (- n 1)) (+ i 1) 'loss))
       (define prev (if (> i 0) (- i 1) 0))
       (values i
               (set next            ;; gradient from next layer (or 'loss)
                    i               ;; own activation
                    prev)))]))     ;; input from previous (clamped to 0)
;; ============================================================
;; Symbolic Memory Optimization
;; ============================================================

;; Create symbolic boolean variables for whether each tensor is kept in memory
(define (create-memory-vars mdl)
  (match mdl
    [(neural-model layers input-shape)
     (define shapes (infer-shapes mdl))
     (define n (length layers))
     
     ;; Create symbolic variables for each activation with unique names
     (define vars
       (for/list ([i (in-range (+ n 1))])
         (define-symbolic* keep boolean?)
         keep))
     
     ;; Return as hash mapping index to symbolic variable
     (for/hash ([i (in-range (+ n 1))]
                [v vars])
       (values i v))]))

;; Generate constraints for correct training
(define (generate-memory-constraints mdl memory-vars)
  (match mdl
    [(neural-model layers input-shape)
     (define n (length layers))
     (define forward-deps (build-forward-dependencies mdl))
     (define backward-deps (build-backward-dependencies mdl))
     
     (define constraints '())
     
     ;; Forward pass constraints
     (for ([i (in-range 1 (+ n 1))])
       (define deps (hash-ref forward-deps i))
       (for ([dep deps])
         (set! constraints
               (cons (=> (hash-ref memory-vars i)
                        (hash-ref memory-vars dep))
                     constraints))))
     
     ;; Backward pass constraints
     (for ([i (in-range n)])
       (define deps (hash-ref backward-deps i))
       (for ([dep deps])
         (when (number? dep)
           (set! constraints
                 (cons (hash-ref memory-vars dep)
                       constraints)))))
     
     ;; Input and output must be kept
     (set! constraints (cons (hash-ref memory-vars 0) constraints))
     (set! constraints (cons (hash-ref memory-vars n) constraints))
     
     (apply && constraints)]))

;; Calculate total memory usage given a solution
(define (calculate-memory-usage mdl memory-solution)
  (match mdl
    [(neural-model layers input-shape)
     (define shapes (infer-shapes mdl))
     
     (for/sum ([i (in-range (length shapes))])
       (if (hash-ref memory-solution i)
           (shape->size (list-ref shapes i))
           0))]))

;; ============================================================
;; Optimization Query
;; ============================================================

(define (optimize-memory-usage mdl max-memory-budget)
  (match mdl
    [(neural-model layers input-shape)
     (define shapes (infer-shapes mdl))
     (displayln (format "Creating memory vars for ~a shapes" (length shapes)))
     
     (define memory-vars (create-memory-vars mdl))
     (displayln (format "Memory vars created: ~a" (hash-keys memory-vars)))
     
     (displayln "Generating constraints...")
     (define constraints (generate-memory-constraints mdl memory-vars))
     
     ;; Calculate symbolic memory usage
     ;; Calculate symbolic memory usage
    (define mem-usage
        (apply +
                (for/list ([i (in-range (length shapes))])
                (if (hash-ref memory-vars i)
                    (shape->size (list-ref shapes i))
                    0))))

     
     (displayln (format "Memory usage expression created, budget: ~a" max-memory-budget))
     
     ;; Try to minimize memory while satisfying constraints
     (define solution
       (synthesize
        #:forall (list)
        #:guarantee (assert (and constraints
                                (<= mem-usage max-memory-budget)))))
     
     (if (sat? solution)
         (let ([evaluated-vars
                (for/hash ([(k v) (in-hash memory-vars)])
                  (values k (evaluate v solution)))])
           (hash 'solution evaluated-vars
                 'memory-usage (calculate-memory-usage mdl evaluated-vars)
                 'shapes shapes))
         'unsat)]))

;; Find minimum memory needed
(define (find-minimum-memory mdl)
  (match mdl
    [(neural-model layers input-shape)
     (define shapes (infer-shapes mdl))
     (define total-memory (for/sum ([s shapes]) (shape->size s)))
     
     (displayln (format "Total possible memory: ~a" total-memory))
     (displayln (format "Shapes: ~a" shapes))
     
     ;; Start with a reasonable budget - try total memory first
     (define result (optimize-memory-usage mdl total-memory))
     
     (if (eq? result 'unsat)
         (displayln "ERROR: Cannot satisfy constraints even with full memory!")
         result)]))

;; Simpler version for testing - just try to fit in total memory
(define (test-memory-optimization mdl)
  (match mdl
    [(neural-model layers input-shape)
     (define shapes (infer-shapes mdl))
     (define total-memory (for/sum ([s shapes]) (shape->size s)))
     
     (displayln (format "\n=== Testing Memory Optimization ==="))
     (displayln (format "Total memory: ~a" total-memory))
     (displayln (format "Shapes: ~a" shapes))
     
     (optimize-memory-usage mdl total-memory)]))

;; ============================================================
;; Pretty Printing
;; ============================================================

(define (print-memory-plan mdl plan)
  (match mdl
    [(neural-model layers input-shape)
     (define solution (hash-ref plan 'solution))
     (define shapes (hash-ref plan 'shapes))
     (define mem-usage (hash-ref plan 'memory-usage))
     
     (displayln "\n=== Memory Optimization Plan ===")
     (displayln (format "Total memory usage: ~a units" mem-usage))
     (displayln "\nLayers kept in memory:")
     
     (for ([i (in-range (length shapes))])
       (when (hash-ref solution i)
         (displayln (format "  Layer ~a: shape ~a, size ~a"
                           i
                           (list-ref shapes i)
                           (shape->size (list-ref shapes i))))))]))

;; ============================================================
;; Example Usage
;; ============================================================

(define mnist-model
  (neural-model
   (list (model-layer (linear 784 128) relu)
         (model-layer (linear 128 64) relu)
         (model-layer (linear 64 10) softmax))
   '(784)))

(displayln "=== Model Parameters ===")
(displayln (format "Total parameters: ~a" (count-model-params mnist-model)))
(displayln (format "Input shape: ~a" (neural-model-input-shape mnist-model)))
(displayln (format "Number of layers: ~a" (length (neural-model-layers mnist-model))))

(displayln "\n=== Testing Memory Optimization ===")
(define opt-plan (test-memory-optimization mnist-model))
(when (not (eq? opt-plan 'unsat))
  (print-memory-plan mnist-model opt-plan))