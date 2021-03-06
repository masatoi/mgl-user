;;; -*- coding:utf-8; mode:lisp; -*-

(in-package :mgl-user)

;;; Create a list of indices suitable as TARGET for ->SOFTMAX-XE-LOSS.
(defun label-target-list (samples)
  (loop for sample in samples
        collect (destructuring-bind (datum &key discard-label-p
                                           sample-visible-p)
                    sample
                  (declare (ignore sample-visible-p))
                  (if discard-label-p
                    nil
                    (datum-label datum)))))

(defmethod set-input (samples (bpn fnn))
  (let* ((inputs (or (find-clump (chunk-lump-name 'inputs nil) bpn :errorp nil)
                     (find-clump 'inputs bpn)))
         (prediction (find-clump 'prediction bpn)))
    (clamp-data samples (nodes inputs))
    (setf (target prediction) (label-target-list samples))))

(defun init-bpn-weights (bpn &key stddev)
  (map-segments (lambda (weight)
                  (cond ((find :scale (name weight))
                         (fill! 1 (nodes weight)))
                        ((or (find :shift (name weight))
                             (find :bias (name weight)))
                         (fill! 0 (nodes weight)))
                        (t
                         (gaussian-random! (nodes weight) :stddev stddev))))
                bpn))

(defun train-fnn (fnn training &key
                                 (n-epochs 3000)
                                 (learning-rate 0.1) (momentum 0.9))
  (let ((optimizer (make-instance 'segmented-gd-optimizer
                      :segmenter
                      (constantly
                       (make-instance 'sgd-optimizer
                          :learning-rate learning-rate
                          :momentum momentum
                          :batch-size (max-n-stripes fnn)))))
        (learner (make-instance 'bp-learner :bpn fnn))
        (dateset (make-sampler training :n-epochs n-epochs)))
    (minimize optimizer learner :dataset dateset)
    fnn))

(defun train-fnn-process (fnn training
                          &key (n-epochs 30) (learning-rate 0.1) (momentum 0.9))
  (with-cuda* ()
    (repeatably ()
      (init-bpn-weights fnn :stddev 0.01)
      (train-fnn fnn training :n-epochs n-epochs :learning-rate learning-rate :momentum momentum)))
  (log-msg "End")
  fnn)

(defun train-fnn-with-monitor (fnn training test
                               &key
                                 (n-epochs 3000)
                                 (learning-rate 0.1) (momentum 0.9))
  (let ((optimizer (monitor-optimization-periodically
                    (make-instance 'segmented-gd-optimizer-with-data
                       :training training
                       :test test
                       :segmenter (constantly
                                   (make-instance 'sgd-optimizer
                                      :learning-rate learning-rate
                                      :momentum momentum
                                      :batch-size (max-n-stripes fnn))))
                    '((:fn log-bpn-test-error :period log-test-period)
                      (:fn reset-optimization-monitors
                       :period log-training-period
                       :last-eval 0))))
        (learner (make-instance 'bp-learner :bpn fnn))
        (dateset (make-sampler training :n-epochs n-epochs)))
    (minimize optimizer learner :dataset dateset)
    fnn))

(defun train-fnn-process-with-monitor (fnn training test
                          &key (n-epochs 30) (learning-rate 0.1) (momentum 0.9))
  (with-cuda* ()
    (repeatably ()
      (init-bpn-weights fnn :stddev 0.01)
      (train-fnn-with-monitor
       fnn training test :n-epochs n-epochs :learning-rate learning-rate :momentum momentum)))
  (log-msg "End")
  fnn)

(defun train-fnn-with-monitor-adam (fnn training test
                               &key
                                 (n-epochs 3000)
                                 (learning-rate 2.e-4) (mean-decay 0.9) (mean-decay-decay (- 1 1.d-7))
                                 (variance-decay 0.999))
  (let ((optimizer (monitor-optimization-periodically
                    (make-instance 'segmented-gd-optimizer-with-data
                       :training training
                       :test test
                       :segmenter (constantly
                                   (make-instance 'adam-optimizer
                                      :learning-rate learning-rate
                                      :mean-decay mean-decay
                                      :mean-decay-decay mean-decay-decay
                                      :variance-decay variance-decay
                                      :batch-size (max-n-stripes fnn))))
                    '((:fn log-bpn-test-error :period log-test-period)
                      (:fn reset-optimization-monitors
                       :period log-training-period
                       :last-eval 0))))
        (learner (make-instance 'bp-learner :bpn fnn))
        (dateset (make-sampler training :n-epochs n-epochs)))
    (minimize optimizer learner :dataset dateset)
    fnn))

(defun train-fnn-process-with-monitor-adam (fnn training test
                          &key (n-epochs 30) (learning-rate 0.1)
                            (mean-decay 0.9) (mean-decay-decay 0.9) (variance-decay 0.9))
  (with-cuda* ()
    (repeatably ()
      (init-bpn-weights fnn :stddev 0.01)
      (train-fnn-with-monitor-adam fnn training test
                                   :n-epochs n-epochs
                                   :learning-rate learning-rate
                                   :mean-decay mean-decay
                                   :mean-decay-decay mean-decay-decay
                                   :variance-decay variance-decay)))
  (log-msg "End")
  fnn)

(defun test-fnn (fnn test)
  (monitor-bpn-results (make-sampler test :max-n (length test))
                       fnn
                       (make-datum-label-monitors fnn)))

(defmethod set-input (samples (bpn fnn))
  (let* ((inputs (or (find-clump (chunk-lump-name 'inputs nil) bpn :errorp nil)
                     (find-clump 'inputs bpn)))
         (prediction (find-clump 'prediction bpn)))
    (clamp-data samples (nodes inputs))
    (setf (target prediction) (label-target-list samples))))

(defun predict-datum (fnn datum)
  (let* ((a (datum-array datum))
         (len (mat-dimension a 0))
         (input-nodes (nodes (find-clump 'inputs fnn)))
         (output-nodes (nodes (find-clump 'prediction fnn))))
    ;; set input
    (loop for i from 0 to (1- len) do
      (setf (mref input-nodes 0 i) (mref a i)))
    ;; run
    (forward fnn)
    ;; return output
    (reshape output-nodes (mat-dimension output-nodes 1))))
