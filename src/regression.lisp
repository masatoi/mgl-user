;;; -*- coding:utf-8; mode:lisp; -*-

(in-package :mgl-user)

;;; Data expression

(defstruct regression-datum
  (id nil :type fixnum)
  (target nil :type mat)
  (array nil :type mat))

;;;; Sampling, clamping, utilities

(defun sample-regression-datum-array (sample)
  (regression-datum-array (first sample)))

(defun clamp-regression-data (samples mat)
  (assert (= (length samples) (mat-dimension mat 0)))
  (map-concat #'copy! samples mat :key #'sample-regression-datum-array))

(defun sample-regression-datum-target (sample)
  (regression-datum-target (first sample)))

(defun clamp-regression-target (samples mat)
  (assert (= (length samples) (mat-dimension mat 0)))
  (map-concat #'copy! samples mat :key #'sample-regression-datum-target))

;;; Regression FNN
(defclass regression-fnn (fnn) ())

(defmethod set-input (samples (bpn regression-fnn))
  (let* ((inputs (find-clump 'inputs bpn))
         (targets (find-clump 'targets bpn)))
    (clamp-regression-data samples (nodes inputs))
    (clamp-regression-target samples (nodes targets))))

;;; Copy dataset
(defun copy-regression-dataset (dataset)
  (let ((new-dataset (map 'vector (lambda (datum) (copy-regression-datum datum)) dataset)))
    (loop for new-datum across new-dataset
          for datum across dataset do
            (setf (regression-datum-array new-datum) (copy-mat (regression-datum-array datum))
                  (regression-datum-target new-datum) (copy-mat (regression-datum-target datum))))
    new-dataset))

;;; Normalize
(defun regression-dataset-average (dataset)
  (let* ((first-datum-array (regression-datum-array (aref dataset 0)))
         (first-datum-target (regression-datum-target (aref dataset 0)))
         (input-dim (mat-dimension first-datum-array 0))
         (output-dim (mat-dimension first-datum-target 0))
         (input-ave (make-mat input-dim))
         (output-ave (make-mat output-dim)))
    (loop for datum across dataset do
      (axpy! (/ 1.0 (length dataset)) (regression-datum-array datum) input-ave)
      (axpy! (/ 1.0 (length dataset)) (regression-datum-target datum) output-ave))
    (values input-ave output-ave)))

(defun regression-dataset-variance (dataset input-ave output-ave)
  (let* ((input-dim   (mat-dimension input-ave 0))
         (output-dim  (mat-dimension output-ave 0))
         (input-var   (make-mat input-dim))
         (output-var  (make-mat output-dim))
         (input-diff  (make-mat input-dim))
         (output-diff (make-mat output-dim)))
    (loop for datum across dataset do
      ;; input
      (copy! input-ave input-diff)
      (axpy! -1.0 (regression-datum-array datum) input-diff)
      (.square! input-diff)
      (axpy! (/ 1.0 (length dataset)) input-diff input-var)
      ;; output
      (copy! output-ave output-diff)
      (axpy! -1.0 (regression-datum-target datum) output-diff)
      (.square! output-diff)
      (axpy! (/ 1.0 (length dataset)) output-diff output-var))
    (values input-var output-var)))

(defun regression-dataset-normalize! (dataset &key test-dataset (noise-degree 1.0))
  (let* ((first-datum-array (regression-datum-array (aref dataset 0)))
         (input-dim (mat-dimension first-datum-array 0))
         (first-datum-target (regression-datum-target (aref dataset 0)))
         (output-dim (mat-dimension first-datum-target 0))
         (input-noise (make-mat input-dim :initial-element noise-degree))
         (output-noise (make-mat output-dim :initial-element noise-degree)))
    (multiple-value-bind (input-ave output-ave)
        (regression-dataset-average dataset)
      (multiple-value-bind (input-var output-var)
          (regression-dataset-variance dataset input-ave output-ave)
        (axpy! 1.0 input-noise input-var)
        (axpy! 1.0 output-noise output-var)
        (.sqrt! input-var)
        (.sqrt! output-var)
        (.inv! input-var)
        (.inv! output-var)
        (flet ((normalize! (datum)
                 (axpy! -1.0 input-ave (regression-datum-array datum))
                 (geem! 1.0 input-var (regression-datum-array datum) 0.0 (regression-datum-array datum))
                 (axpy! -1.0 output-ave (regression-datum-target datum))
                 (geem! 1.0 output-var (regression-datum-target datum) 0.0 (regression-datum-target datum))))
          (loop for datum across dataset do (normalize! datum))
          (if test-dataset
            (loop for datum across test-dataset do (normalize! datum)))
          'done)))))

;;;; Activation access utilities
(defun activations-output (activations)
  (aref (clumps activations) 3))

(defun find-last-activation (bpn)
  (let ((clumps-vec (clumps bpn)))
    (loop for i from (1- (length clumps-vec)) downto 0 do
      (let ((clump (aref clumps-vec i)))
      (typecase clump
        (->activation (return clump)))))))

;;; Monitoring
(defun log-regression-cost (optimizer learner)
  (when (zerop (n-instances optimizer))
    (report-optimization-parameters optimizer learner))
  (log-msg "train/test at n-instances: ~S (~A ephochs)~%" (n-instances optimizer)
           (/ (n-instances optimizer) (length (training optimizer))))
  (log-padded
   (let ((bpn (bpn learner))
         (monitors (monitors learner)))
     (append
      (monitor-bpn-results (make-sampler (training optimizer) :max-n 10000) bpn (list (car monitors)))
      (if (test optimizer)
        (monitor-bpn-results (make-sampler (test optimizer)) bpn (cdr monitors))))))
  (log-mat-room)
  (log-msg "---------------------------------------------------~%"))

(defun train-regression-fnn-with-monitor
    (fnn training &key test (n-epochs 3000) (learning-rate 0.1) (momentum 0.9))
  (let* ((optimizer (monitor-optimization-periodically
                     (make-instance 'segmented-gd-optimizer-with-data
                        :training training :test test
                        :segmenter (constantly
                                    (make-instance 'sgd-optimizer
                                       :learning-rate learning-rate
                                       :momentum momentum
                                       :batch-size (max-n-stripes fnn))))
                     `((:fn log-regression-cost :period ,(length training))
                       (:fn reset-optimization-monitors
                            :period ,(length training)
                            :last-eval 0))))
         (measurer (lambda (instances bpn)
                     (declare (ignore instances))
                     (mgl-bp::cost bpn)))
         (monitors (cons (make-instance 'monitor
                            :measurer measurer
                            :counter (make-instance 'rmse-counter
                                        :prepend-attributes '(:event "rmse." :dataset "train")))
                         (if test
                           (list (make-instance 'monitor
                                    :measurer measurer
                                    :counter (make-instance 'rmse-counter
                                                :prepend-attributes '(:event "rmse." :dataset "test")))))))
         (learner (make-instance 'bp-learner :bpn fnn :monitors monitors))
         (dateset (make-sampler training :n-epochs n-epochs)))
    (minimize optimizer learner :dataset dateset)
    fnn))

(defun train-regression-fnn-process-with-monitor
    (fnn training &key test (n-epochs 30) (learning-rate 0.1) (momentum 0.9) without-initialize)
  (with-cuda* ()
    (repeatably ()
      (if (null without-initialize)
        (init-bpn-weights fnn :stddev 0.01))
      (train-regression-fnn-with-monitor
       fnn training :test test :n-epochs n-epochs :learning-rate learning-rate :momentum momentum)))
  (log-msg "End")
  fnn)

;;; L2-normalizing

(defclass bpn-gd-optimizer (segmented-gd-optimizer) ())

(defclass bpn-gd-segment-optimizer (sgd-optimizer)
  ((n-instances-in-epoch
    :initarg :n-instances-in-epoch
    :reader n-instances-in-epoch)
   (n-epochs-to-reach-final-momentum
    :initarg :n-epochs-to-reach-final-momentum
    :reader n-epochs-to-reach-final-momentum)
   (learning-rate-decay
    :initform 0.998
    :initarg :learning-rate-decay
    :accessor learning-rate-decay)))

(defun make-grouped-segmenter (group-name-fn segmenter)
  (let ((group-name-to-optimizer (make-hash-table :test #'equal)))
    (lambda (segment)
      (let ((group-name (funcall group-name-fn segment)))
        (or (gethash group-name group-name-to-optimizer)
            (setf (gethash group-name group-name-to-optimizer)
                  (funcall segmenter segment)))))))

(defun weight-lump-target-name (lump)
  (let ((name (name lump)))
    (assert (listp name))
    (assert (= 2 (length name)))
    (if (eq (first name) :cloud)
        (second (second name))
        (second name))))

(defun make-dwim-grouped-segmenter (segmenter)
  (make-grouped-segmenter #'weight-lump-target-name segmenter))

(defmethod learning-rate ((optimizer bpn-gd-segment-optimizer))
  (* (expt (learning-rate-decay optimizer)
           (/ (n-instances optimizer)
              (n-instances-in-epoch optimizer)))
     (- 1 (momentum optimizer))
     (slot-value optimizer 'learning-rate)))

(defmethod momentum ((optimizer bpn-gd-segment-optimizer))
  (let ((n-epochs-to-reach-final (n-epochs-to-reach-final-momentum optimizer))
        (initial 0.5)
        (final 0.99)
        (epoch (/ (n-instances optimizer) (n-instances-in-epoch optimizer))))
    (if (< epoch n-epochs-to-reach-final)
        (let ((weight (/ epoch n-epochs-to-reach-final)))
          (+ (* initial (- 1 weight))
             (* final weight)))
        final)))

(defun train-bpn-gd (bpn training
                     &key test (n-epochs 200) l2-upper-bound learning-rate learning-rate-decay
                       input-weight-penalty)
  (flet ((make-optimizer (lump)
           (let ((optimizer (make-instance 'bpn-gd-segment-optimizer
                               :n-instances-in-epoch (length training)
                               :n-epochs-to-reach-final-momentum (min 500 (/ n-epochs 2))
                               :learning-rate learning-rate
                               :learning-rate-decay learning-rate-decay
                               :weight-penalty (if (and input-weight-penalty
                                                        (member (name lump) '((inputs f1)) :test #'name=))
                                                 input-weight-penalty
                                                 0)
                               :batch-size (max-n-stripes bpn))))
             (when l2-upper-bound
               (arrange-for-renormalizing-activations bpn optimizer l2-upper-bound))
             optimizer))
         (make-segmenter (fn)
           (let ((dwim (make-dwim-grouped-segmenter fn)))
             (lambda (lump)
               (if (and l2-upper-bound
                        (not (and input-weight-penalty
                                  (member (name lump) '((inputs f1) (:bias f1))
                                          :test #'name=))))
                 (funcall dwim lump)
                 (funcall fn lump))))))
    (let* ((optimizer (monitor-optimization-periodically
                       (make-instance 'segmented-gd-optimizer-with-data
                          :training training :test test
                          :segmenter (make-segmenter #'make-optimizer))
                       `((:fn log-regression-cost :period ,(length training))
                         (:fn reset-optimization-monitors
                              :period ,(length training)
                              :last-eval 0))))
           (measurer (lambda (instances bpn)
                     (declare (ignore instances))
                     (mgl-bp::cost bpn)))
           (monitors (cons (make-instance 'monitor
                              :measurer measurer
                              :counter (make-instance 'rmse-counter
                                          :prepend-attributes '(:event "rmse." :dataset "train")))
                           (if test
                             (list (make-instance 'monitor
                                      :measurer measurer
                                      :counter (make-instance 'rmse-counter
                                                  :prepend-attributes '(:event "rmse." :dataset "test")))))))
           (learner (make-instance 'bp-learner :bpn bpn :monitors monitors))
           (dataset (make-sampler training :n-epochs n-epochs)))
      (log-msg "Starting to train the whole BPN~%")
      (minimize optimizer learner :dataset dataset))))

(defun train-bpn-gd-process
    (fnn training &key test (n-epochs 30)
                    (l2-upper-bound 1.9364917) (learning-rate 1) (learning-rate-decay 0.996)
                    input-weight-penalty without-initialize)
  (with-cuda* ()
    (repeatably ()
      (if (null without-initialize)
        (init-bpn-weights fnn :stddev 0.01))
      (train-bpn-gd fnn training
                    :test test
                    :n-epochs n-epochs :l2-upper-bound l2-upper-bound
                    :learning-rate learning-rate :learning-rate-decay learning-rate-decay 
                    :input-weight-penalty input-weight-penalty)))
  (log-msg "End")
  fnn)

;; (train-bpn-gd-process fnn-regression *casp-dataset-normal* :input-weight-penalty 0.000001)
;; (train-bpn-gd-process fnn-maxout-dropout-regression *casp-dataset-normal* :input-weight-penalty 0.000001)

;;; Prediction
(defun predict-regression-datum (fnn regression-datum)
  (let* ((a (regression-datum-array regression-datum))
         (len (mat-dimension a 0))
         (input-nodes (nodes (find-clump 'inputs fnn)))
         (output-nodes (nodes (activations-output (find-last-activation fnn)))))
    ;; set input
    (loop for i from 0 to (1- len) do
      (setf (mref input-nodes 0 i) (mref a i)))
    ;; run
    (forward fnn)
    ;; return output
    (reshape output-nodes (mat-dimension output-nodes 1))))

(defun array-to-64x64-array (arr)
  (let ((a (make-array '(64 64))))
    (loop for i from 0 to 63 do
      (loop for j from 0 to 63 do
        (setf (aref a i j) (aref arr (+ (* i 64) j)))))
    a))
