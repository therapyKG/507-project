#lang rosette

(require racket/match)
(require racket/format)
(require "syntax.rkt")

(provide pipeline)

;; ============================================================
;; 1. Concrete Layer Analysis (Helper)
;; ============================================================

(define (count-layer-params lyr)
  (match lyr
    [(layer 'linear (list in out) _) (+ (* in out) out)]
    [(layer 'conv (list in out k s p) _) (+ (* in out k k) out)]
    [_ 0]))

(define (validate-layer lyr input-shape)
  (match* (lyr input-shape)
    [((layer 'linear (list in out) _) (list curr-dim))
     (if (= in curr-dim) (values (list out) #f)
         (values (list out) (format "Dim Mismatch: Linear expects ~a, got ~a" in curr-dim)))]
    [((layer 'linear _ _) other) (values '(0) (format "Linear expects 1D, got ~a" other))]
    
    ;; FIX: Added (+ ... 1) to the dimension calculation
    [((layer 'conv (list in out k s p) _) (list c h w))
     (if (= in c)
         (values (list out 
                       (+ (quotient (+ h (* 2 p) (- k)) s) 1) 
                       (+ (quotient (+ w (* 2 p) (- k)) s) 1)) 
                 #f)
         (values (list out 1 1) (format "Channel Mismatch: Conv expects ~a, got ~a" in c)))]
    [((layer 'conv _ _) other) (values '(0) (format "Conv expects 3D, got ~a" other))]
     
    [((layer 'maxpool (list k s) _) (list c h w))
     (values (list c (quotient h s) (quotient w s)) #f)]
    [((layer 'maxpool _ _) other) (values '(0) "Maxpool expects 3D")]

    [((layer 'flatten _ _) shape) (if (list? shape) (values (list (apply * shape)) #f) (values '(1) "Invalid flatten"))]
    [(_ _) (values '(0) "Unknown layer")]))

;; ============================================================
;; 2. Job Analysis (Returns memory coefficients)
;; ============================================================

(struct job-metrics (id name static-bytes per-sample-bytes max-count) #:transparent)

(define (analyze-job job index)
  (define mdl (train-job-model job))
  (define data (train-job-dataset job))
  
  (define input-shape (dataset-struct-shape data))
  (define layers (model-struct-layers mdl))
  (define model-bytes (/ (model-struct-bit-width mdl) 8.0))
  (define data-bytes (/ (dataset-struct-bit-width data) 8.0))
  
  ;; 1. Trace Architecture
  (define total-params 0)
  (define input-elements (if (empty? input-shape) 0 (apply * input-shape)))
  (define total-activation-elements 0)
  
  (define-values (final-shape error-count)
    (for/fold ([current-shape input-shape] [err-count 0])
              ([l layers] [i (in-naturals 1)])
      (set! total-params (+ total-params (count-layer-params l)))
      (define-values (next-shape error-msg) (validate-layer l current-shape))
      (unless (eq? (layer-type l) 'flatten)
        (set! total-activation-elements (+ total-activation-elements (apply * next-shape))))
      (when error-msg (printf "    [Job ~a ERROR] Layer ~a: ~a\n" index i error-msg))
      (values next-shape (if error-msg (+ err-count 1) err-count))))

  (if (> error-count 0)
      #f 
      (job-metrics
       index
       (dataset-struct-spec data)
       (inexact->exact (round (* total-params model-bytes 4))) ;; Static
       (inexact->exact (round (+ (* total-activation-elements model-bytes 2) (* input-elements data-bytes)))) ;; Per Sample
       (dataset-struct-count data))))

;; ============================================================
;; 3. Partition Solver (Optimizes a specific group)
;; ============================================================

(define (solve-partition partition-id group-metrics vram-limit-bytes)
  (printf "\n  [Partition ~a]: Executing ~a jobs concurrently\n" 
          partition-id (length group-metrics))
  
  (define sym-batches 
    (for/list ([m group-metrics]) 
      (define-symbolic* b integer?) 
      b))
  
  (define (memory-cost idx batch-var)
    (define m (list-ref group-metrics idx))
    (+ (job-metrics-static-bytes m) 
       (* (job-metrics-per-sample-bytes m) batch-var)))

  (define total-memory-usage (apply + (for/list ([b sym-batches] [i (in-naturals)]) (memory-cost i b))))

  (define constraints
    (and 
     (<= total-memory-usage vram-limit-bytes)
     (apply && (for/list ([b sym-batches] [m group-metrics])
                  (and (>= b 1) (<= b (job-metrics-max-count m)))))))

  (define sum-batches (apply + sym-batches))
  (define solution 
    (optimize #:maximize (list sum-batches)
              #:guarantee (assert constraints)))
  
  (if (unsat? solution)
      (begin (printf "    FAILED: Could not find valid batch sizes.\n") #f)
      (begin
        (define partition-mem 0)
        (for ([b sym-batches] [m group-metrics])
          (define val (evaluate b solution))
          (define mem (+ (job-metrics-static-bytes m) (* (job-metrics-per-sample-bytes m) val)))
          (set! partition-mem (+ partition-mem mem))
          (printf "    - Job ~a (~a): Batch ~a | Mem ~a MB\n" 
                  (job-metrics-id m) (job-metrics-name m) val (~r (/ mem 1024.0 1024.0) #:precision 2)))
        (printf "    >> Partition VRAM Usage: ~a MB\n" (~r (/ partition-mem 1024.0 1024.0) #:precision 2))
        #t)))

;; ============================================================
;; 4. Pipeline Orchestrator (Greedy Partitioning)
;; ============================================================

(define (solve-pipeline jobs config)
  (define vram-limit-mb (pipeline-config-vram-mb config))
  (define vram-limit-bytes (inexact->exact (round (* vram-limit-mb 1024 1024))))

  (printf "\n=== Pipeline Memory Planning ===\n")
  (printf "Global VRAM Budget: ~a MB\n" vram-limit-mb)
  
  ;; 1. Analyze all jobs first
  (define all-metrics
    (for/list ([j jobs] [i (in-naturals 1)])
      (analyze-job j i)))

  (cond
    [(member #f all-metrics) 
     (printf "Status: FAILED (Architecture Errors)\n") #f]
    [else
     ;; 2. Partition Strategy Loop
     (define partitions '())
     (define current-group '())
     (define current-min-cost 0)
     (define failure #f)

     (for ([m all-metrics])
       #:break failure
       (define job-min-cost (+ (job-metrics-static-bytes m) (job-metrics-per-sample-bytes m)))
       
       (cond
         ;; Case A: Single job is too big for VRAM
         [(> job-min-cost vram-limit-bytes)
          (printf "\nCRITICAL ERROR: Job ~a (~a) requires ~a MB (min), which exceeds VRAM limit.\n"
                  (job-metrics-id m) (job-metrics-name m) (~r (/ job-min-cost 1024.0 1024.0) #:precision 2))
          (set! failure #t)]
         
         ;; Case B: Job fits in current group
         [(<= (+ current-min-cost job-min-cost) vram-limit-bytes)
          (set! current-group (append current-group (list m)))
          (set! current-min-cost (+ current-min-cost job-min-cost))]
         
         ;; Case C: Job doesn't fit -> Commit current group, start new one
         [else
          (set! partitions (append partitions (list current-group)))
          (set! current-group (list m))
          (set! current-min-cost job-min-cost)]))
     
     ;; Add remaining group
     (unless (empty? current-group)
       (set! partitions (append partitions (list current-group))))

     (if failure
         #f
         (begin
           (printf "\nStrategy: Split pipeline into ~a sequential partitions.\n" (length partitions))
           (define all-ok #t)
           (for ([p partitions] [i (in-naturals 1)])
             (unless (solve-partition i p vram-limit-bytes)
               (set! all-ok #f)))
           
           (if all-ok
               (printf "\nStatus: SUCCESS (Plan Generated)\n")
               (printf "\nStatus: FAILED (Solver Error)\n"))))]))

;; ============================================================
;; 5. Pipeline Macro
;; ============================================================

(define (parse-pipeline-args args)
  (define with-clause (findf (lambda (x) (and (list? x) (eq? (car x) 'with))) args))
  (define jobs (filter train-job? args))
  (define config 
    (if with-clause
        (match (cadr with-clause)
          [(list (list _ v)) (pipeline-config v)] ;; Match any key for VRAM
          [_ (error "Invalid pipeline config. Expected ((vram X))")])
        (pipeline-config 100000)))
  (values jobs config))

(define-syntax pipeline
  (syntax-rules (with)
    ;; Case 1: Match pipeline jobs ... with ((v limit))
    ;; We use 'v' as a wildcard variable here to accept 'vram'
    [(_ item ... with ((v limit)))
     (solve-pipeline (list item ...) (pipeline-config limit))]
    
    ;; Case 2: No config provided
    [(_ item ...)
     (solve-pipeline (list item ...) (pipeline-config 100000))]))