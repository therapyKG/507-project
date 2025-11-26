#lang rosette

(require racket/match)
(require racket/format)
(require "syntax.rkt")

(provide analyze-training train)

;; ============================================================
;; 1. Parameter Counting & Layer Logic (Concrete Helpers)
;; ============================================================
;; These remain concrete for now as they deal with fixed model definitions.

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

;; ============================================================
;; 2. Main Analysis Logic (Symbolic)
;; ============================================================

(define (analyze-training mdl data [config #f])
  (define input-shape (dataset-struct-shape data))
  (define layers (model-struct-layers mdl))
  
  ;; Precision configuration
  (define data-bits (dataset-struct-bit-width data))
  (define model-bits (model-struct-bit-width mdl))
  (define model-bytes (/ model-bits 8.0))
  (define data-bytes (/ data-bits 8.0))
  
  ;; Config defaults
  (define vram-limit-mb (if config (train-config-vram-mb config) 100000))
  (define user-requested-batch (if config (train-config-batch-size config) 1))
  
  (printf "\n=== Training Analysis (Symbolic Engine) ===\n")
  (printf "Dataset: ~a | Count: ~a | Input: ~a | Precision: ~a-bit\n" 
          (dataset-struct-spec data) (dataset-struct-count data) input-shape data-bits)
  (printf "Model Precision: ~a-bit (~a bytes/param)\n" model-bits model-bytes)
  
  (when config
    (printf "User Constraint: Batch Size ~a | VRAM Limit ~a MB\n" user-requested-batch vram-limit-mb))

  ;; ----------------------------------------------------------
  ;; Step 1: Concrete Architecture Analysis
  ;; (We still partial-evaluate the architecture structure concretely)
  ;; ----------------------------------------------------------
  
  (define total-params 0)
  (define input-elements (if (empty? input-shape) 0 (apply * input-shape)))
  (define input-mem-per-sample (* input-elements data-bytes))
  (define total-activation-elements 0)
  
  (define-values (final-shape error-count)
    (for/fold ([current-shape input-shape] [err-count 0])
              ([l layers] [i (in-naturals 1)])
      
      (set! total-params (+ total-params (count-layer-params l)))
      (define-values (next-shape error-msg) (validate-layer l current-shape))
      
      (define layer-output-elements (apply * next-shape))
      (unless (eq? (layer-type l) 'flatten)
        (set! total-activation-elements (+ total-activation-elements layer-output-elements)))
      
      (printf "Layer ~a (~a): ~a -> ~a" i (layer-type l) current-shape next-shape)
      (when error-msg (printf " [ERROR: ~a]" error-msg))
      (newline)
      
      (values next-shape (if error-msg (+ err-count 1) err-count))))

  ;; ----------------------------------------------------------
  ;; Step 2: Symbolic Memory Planning
  ;; ----------------------------------------------------------
  
  ;; 1. Define Constants
  ;; Use exact integers for bytes to ensure solver stability
  (define vram-limit-bytes (inexact->exact (round (* vram-limit-mb 1024 1024))))
  (define static-mem-bytes (inexact->exact (round (* total-params model-bytes 4))))
  (define per-sample-bytes (inexact->exact (round (+ (* total-activation-elements model-bytes 2) 
                                                     input-mem-per-sample))))
  
  ;; 2. Define Symbolic Variable
  (define-symbolic* sym-batch integer?)
  
  ;; 3. Formulate Constraints
  (define memory-constraint 
    (<= (+ static-mem-bytes (* per-sample-bytes sym-batch)) vram-limit-bytes))
    
  (define logical-constraints
    (and (> sym-batch 0)
         (<= sym-batch (dataset-struct-count data))))
         
  (define all-constraints (and memory-constraint logical-constraints))

  ;; 4. Solve: Check User Request
  ;; We ask: "Is it possible for sym-batch to equal the user request AND satisfy constraints?"
  (define user-check-solution
    (solve (assert (and all-constraints (= sym-batch user-requested-batch)))))
    
  (define user-request-valid? (sat? user-check-solution))

  ;; 5. Optimize: Find Max Possible Batch
  ;; We ask: "What is the maximum value of sym-batch that satisfies constraints?"
  (define max-batch-solution
    (optimize #:maximize (list sym-batch)
              #:guarantee (assert all-constraints)))
              
  (define max-possible-batch
    (if (sat? max-batch-solution)
        (evaluate sym-batch max-batch-solution)
        0))

  ;; ----------------------------------------------------------
  ;; Step 3: Reporting
  ;; ----------------------------------------------------------

  (define requested-batch-mem (+ static-mem-bytes (* per-sample-bytes user-requested-batch)))
  (define requested-batch-mb (/ requested-batch-mem 1024.0 1024.0))

  (printf "-------------------------\n")
  (printf "Memory Analysis (Solved by Rosette):\n")
  (printf "  Static Model Memory: ~a MB\n" (~r (/ static-mem-bytes 1024.0 1024.0) #:precision 2))
  (printf "  Memory Per Sample:   ~a KB\n" (~r (/ per-sample-bytes 1024.0) #:precision 2))
  (printf "  \n")
  (printf "  Max Batch (Solver):  ~a\n" max-possible-batch)
  (printf "  Requested Batch:     ~a\n" user-requested-batch)
  (printf "  Total Estimated VRAM:~a MB / ~a MB\n" 
          (~r requested-batch-mb #:precision 2) vram-limit-mb)
  
  (printf "=========================\n")
  (cond
    [(> error-count 0)
     (printf "Status: FAILED (Architecture Errors)\n") #f]
    [(not user-request-valid?)
     (printf "Status: FAILED (OOM - Solver could not fit batch ~a. Max is ~a)\n" 
             user-requested-batch max-possible-batch) #f]
    [else
     (printf "Status: SUCCESS\n") #t]))

;; ============================================================
;; 3. Macro Definition
;; ============================================================

(define-syntax train
  (syntax-rules (on with vram batch)
    [(_ model-expr on dataset-expr with ((vram v-val) (batch b-val)))
     (analyze-training model-expr dataset-expr (train-config v-val b-val))]
    [(_ model-expr on dataset-expr)
     (analyze-training model-expr dataset-expr #f)]))