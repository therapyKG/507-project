#lang rosette

(require racket/match)
(require racket/format)
(require "syntax.rkt")

(provide pipeline)

;; ============================================================
;; 1. Concrete Layer Analysis (Helpers)
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
;; 2. Job Analysis
;; ============================================================

;; Updated struct to store separated memory factors
(struct job-metrics (id name models static-bytes act-bytes-per-sample input-bytes-per-sample max-count) #:transparent)

(define (analyze-job job index)
  (define models (train-job-models job)) 
  (define data (train-job-dataset job))
  
  (define input-shape (dataset-struct-shape data))
  (define data-bytes (/ (dataset-struct-bit-width data) 8.0))

  (printf "  - Job ~a: ~a on ~a\n" index (dataset-struct-spec data) input-shape)

  (define total-params 0)
  (define total-activation-elements 0)
  (define input-elements (if (empty? input-shape) 0 (apply * input-shape)))
  
  (define-values (final-shape total-error-count)
    (for/fold ([current-shape input-shape] [err-acc 0])
              ([m models] [m-idx (in-naturals 1)])
      
      (define layers (model-struct-layers m))
      (define-values (m-out-shape m-errs)
        (for/fold ([c-shape current-shape] [e-acc 0])
                  ([l layers] [l-idx (in-naturals 1)])
          (set! total-params (+ total-params (count-layer-params l)))
          (define-values (next-shape error-msg) (validate-layer l c-shape))
          (unless (eq? (layer-type l) 'flatten)
            (set! total-activation-elements (+ total-activation-elements (apply * next-shape))))
          (when error-msg (printf "    [Job ~a Model ~a Layer ~a] ERROR: ~a\n" index m-idx l-idx error-msg))
          (values next-shape (if error-msg (+ e-acc 1) e-acc))))
      (values m-out-shape (+ err-acc m-errs))))

  (if (> total-error-count 0)
      #f
      (job-metrics
       index
       (dataset-struct-spec data)
       models
       0 ;; Placeholder for aggregated static (unused in liveness solver)
       ;; Activations: Total elements * 4 bytes (float32) * 2 (fwd+bwd)
       (inexact->exact (round (* total-activation-elements 4 2))) 
       ;; Input Data: Elements * data-width
       (inexact->exact (round (* input-elements data-bytes)))
       (dataset-struct-count data))))

(define (get-model-static-bytes m)
  (define b (/ (model-struct-bit-width m) 8.0))
  (define params (for/sum ([l (model-struct-layers m)]) (count-layer-params l)))
  (inexact->exact (round (* params b 4))))

;; ============================================================
;; 3. Liveness-Aware Solver & Reporting
;; ============================================================

(define (solve-pipeline jobs config)
  (define vram-limit-mb (pipeline-config-vram-mb config))
  (define vram-limit-bytes (inexact->exact (round (* vram-limit-mb 1024 1024))))

  (printf "\n=== Pipeline Memory Planning ===\n")
  (printf "Global VRAM Budget: ~a MB\n" vram-limit-mb)
  
  (define metrics-list
    (for/list ([j jobs] [i (in-naturals 1)])
      (analyze-job j i)))

  (cond
    [(member #f metrics-list) 
     (printf "Status: FAILED (Architecture Errors)\n") #f]
    [else
     ;; Component Identification
     (define all-models (remove-duplicates (apply append (map job-metrics-models metrics-list)) eq?))
     (define model-ids (for/list ([m all-models] [i (in-naturals)]) (cons m i)))
     (define (get-id m) (cdr (assq m model-ids)))
     (define (get-comp-size id) 
       (define m (car (findf (lambda (p) (= (cdr p) id)) model-ids)))
       (get-model-static-bytes m))
     
     (define (ids->names ids)
       (for/list ([id ids])
         (define m (car (findf (lambda (p) (= (cdr p) id)) model-ids)))
         (format "~a:~a" id (model-struct-name m))))

     (define component-info
       (for/list ([m all-models])
         (define id (get-id m))
         (define size (get-model-static-bytes m))
         (define used-in-indices
           (for/list ([j metrics-list] [idx (in-naturals 0)]
                      #:when (member m (job-metrics-models j)))
             idx))
         (list m id size (first used-in-indices) (last used-in-indices))))

     (printf "\nDetected ~a Unique Components:\n" (length all-models))
     (for ([c component-info])
       (match-define (list m id size start end) c)
       (printf "  [C~a] ~a (uid:~a): ~a MB | Jobs ~a -> ~a\n" 
               id (model-struct-name m) (model-struct-uid m) 
               (~r (/ size 1024.0 1024.0) #:precision 2) (+ 1 start) (+ 1 end)))

     ;; Symbolic Setup
     (define sym-batches 
       (for/list ([m metrics-list]) (define-symbolic* b integer?) b))

     (printf "\nGenerating Memory Plan...\n")
     
     (define constraints
       (apply &&
        (for/list ([j-metric metrics-list] [j-idx (in-naturals)] [batch-var sym-batches])
          (define required-comps 
            (filter (lambda (c) (member (first c) (job-metrics-models j-metric))) component-info))
          
          (define required-mem (apply + (map third required-comps)))
          ;; Updated to use sum of activations + inputs
          (define dynamic-mem-expr 
            (* (+ (job-metrics-act-bytes-per-sample j-metric) 
                  (job-metrics-input-bytes-per-sample j-metric)) 
               batch-var))
          
          (and (>= batch-var 1)
               (<= batch-var (job-metrics-max-count j-metric))
               (<= (+ required-mem dynamic-mem-expr) vram-limit-bytes)))))

     ;; Solve
     (define sum-batches (apply + sym-batches))
     (define solution (optimize #:maximize (list sum-batches) #:guarantee (assert constraints)))

     (if (unsat? solution)
         (begin (printf "Status: FAILED (OOM - Cannot fit basic requirements)\n") #f)
         (begin
           (printf "\nStatus: SUCCESS\n")
           (printf "------------------------------------------------------------\n")
           
           (define resident-set '()) 
           
           (for ([j-idx (in-naturals)] [j-metric metrics-list] [batch-sym sym-batches])
             (define val (evaluate batch-sym solution))
             
             (define required-indices (map get-id (job-metrics-models j-metric)))
             
             (define idle-candidates
               (filter (lambda (c) 
                         (match-define (list _ idx _ start end) c)
                         (and (<= start j-idx end) (not (member idx required-indices))))
                       component-info))
             
             (define req-mem (apply + (map get-comp-size required-indices)))
             
             ;; Calculate broken down memory costs
             (define act-mem (* (job-metrics-act-bytes-per-sample j-metric) val))
             (define inp-mem (* (job-metrics-input-bytes-per-sample j-metric) val))
             (define dyn-mem (+ act-mem inp-mem))
             
             (define available-for-cache (- vram-limit-bytes (+ req-mem dyn-mem)))
             
             (define kept-cached-indices '())
             (define current-cache-fill 0)
             
             (for ([c idle-candidates])
               (match-define (list _ idx size _ _) c)
               (when (<= (+ current-cache-fill size) available-for-cache)
                 (set! kept-cached-indices (cons idx kept-cached-indices))
                 (set! current-cache-fill (+ current-cache-fill size))))
             
             (define active-set (append required-indices kept-cached-indices))
             
             (define loaded (filter (lambda (x) (not (member x resident-set))) required-indices))
             (define held-active (filter (lambda (x) (member x resident-set)) required-indices))
             (define held-cached (filter (lambda (x) (member x resident-set)) kept-cached-indices))
             (define evicted (filter (lambda (x) (and (member x resident-set) 
                                                      (not (member x active-set))
                                                      (member x (map second idle-candidates)))) 
                                     resident-set))
             (define freed (filter (lambda (x) (and (member x resident-set) 
                                                    (not (member x active-set))
                                                    (not (member x (map second idle-candidates))))) 
                                   resident-set))

             (printf "Job ~a (~a): Batch ~a\n" (+ 1 j-idx) (job-metrics-name j-metric) val)
             
             (unless (empty? loaded)      (printf "  [+] LOADS:   ~a\n" (ids->names loaded)))
             (unless (empty? held-active) (printf "  [=] HOLDS:   ~a (Active)\n" (ids->names held-active)))
             (unless (empty? held-cached) (printf "  [=] CACHES:  ~a (Idle)\n" (ids->names held-cached)))
             (unless (empty? evicted)     (printf "  [-] EVICTS:  ~a (To save space)\n" (ids->names evicted)))
             (unless (empty? freed)       (printf "  [-] FREES:   ~a (Lifespan ended)\n" (ids->names freed)))
             
             (printf "  - Memory Breakdown:\n")
             (printf "    * Weights:     ~a MB\n" (~r (/ req-mem 1024.0 1024.0) #:precision 2))
             (printf "    * Activations: ~a MB\n" (~r (/ act-mem 1024.0 1024.0) #:precision 2))
             (printf "    * Dataset:     ~a MB\n" (~r (/ inp-mem 1024.0 1024.0) #:precision 2))
             (when (> current-cache-fill 0)
                (printf "    * Cached:      ~a MB\n" (~r (/ current-cache-fill 1024.0 1024.0) #:precision 2)))

             (printf "  >>> VRAM: ~a MB / ~a MB\n" 
                     (~r (/ (+ req-mem dyn-mem current-cache-fill) 1024.0 1024.0) #:precision 2) vram-limit-mb)
             (printf "------------------------------------------------------------\n")
             (set! resident-set active-set))
           #t))]))

;; ============================================================
;; 4. Pipeline Macro
;; ============================================================

(define (parse-pipeline-args args)
  (define with-clause (findf (lambda (x) (and (list? x) (eq? (car x) 'with))) args))
  (define jobs (filter train-job? args))
  (define config 
    (if with-clause
        (match (cadr with-clause)
          [(list (list _ v)) (pipeline-config v)] 
          [_ (error "Invalid pipeline config. Expected ((vram X))")])
        (pipeline-config 100000)))
  (values jobs config))

(define-syntax pipeline
  (syntax-rules (with)
    [(_ item ... with ((v limit)))
     (solve-pipeline (list item ...) (pipeline-config limit))]
    [(_ item ...)
     (solve-pipeline (list item ...) (pipeline-config 100000))]))