;;; -*- coding:utf-8; mode:lisp -*-

(in-package :mgl-user)

(defclass segmented-gd-optimizer-with-data (segmented-gd-optimizer)
  ((training :initarg :training :reader training)
   (test :initarg :test :reader test)))

(defun log-training-period (optimizer learner)
  (declare (ignore learner))
  (length (training optimizer)))

(defun log-test-period (optimizer learner)
  (declare (ignore learner))
  (length (training optimizer)))

(defun sample-datum-label-index (sample)
  (datum-label (car sample)))

(defun sample-datum-label-index-distribution (sample)
  (let ((d (make-array 10 :initial-element 0)))
    (setf (aref d (datum-label (car sample))) 1)
    d))

(defun make-datum-label-monitors (model &key attributes)
  (make-label-monitors
   model
   :label-index-fn #'sample-datum-label-index
   :label-index-distribution-fn #'sample-datum-label-index-distribution
   :attributes attributes))

(defun log-bpn-test-error (optimizer learner)
  (when (zerop (n-instances optimizer))
    (report-optimization-parameters optimizer learner))
  (log-msg "test at n-instances: ~S~%" (n-instances optimizer))
  (log-padded
   (let ((bpn (bpn learner)))
     (append
      (monitor-bpn-results (make-sampler (training optimizer)
                                         :max-n 10000)
                           bpn
                           (make-datum-label-monitors
                            bpn :attributes '(:event "pred."
                                              :dataset "train")))
      (monitor-bpn-results (make-sampler (test optimizer)) bpn
                           (make-datum-label-monitors
                            bpn :attributes '(:event "pred."
                                              :dataset "test"))))))
  (log-mat-room)
  (log-msg "---------------------------------------------------~%"))

(defun monitor-rbm-cesc-accuracy (rbm sampler attributes)
  (if (dbn rbm)
      (monitor-dbn-mean-field-reconstructions
       sampler (dbn rbm)
       (make-datum-label-monitors (dbn rbm) :attributes attributes)
       :set-visible-p t)
      (monitor-bm-mean-field-reconstructions
       sampler rbm
       (make-datum-label-monitors rbm :attributes attributes)
       :set-visible-p t)))

(defun log-rbm-test-error (optimizer learner)
  (when (zerop (n-instances optimizer))
    (report-optimization-parameters optimizer learner))
  (log-msg "test at n-instances: ~S~%" (n-instances optimizer))
  (let ((rbm (rbm learner)))
    (log-padded
     (append
      (monitor-rbm-cesc-accuracy rbm (make-tiny-sampler (training optimizer))
                                 '(:event "pred." :dataset "train+"))
      (monitor-rbm-cesc-accuracy rbm (make-tiny-sampler (training optimizer)
                                                        :discard-label-p t)
                                 '(:event "pred." :dataset "train"))
      (monitor-dbn-mean-field-reconstructions
       (make-sampler (test optimizer)) (dbn rbm)
       (make-reconstruction-monitors
        (dbn rbm) :attributes '(:event "pred." :dataset "test+")))
      (monitor-rbm-cesc-accuracy rbm (make-sampler (test optimizer)
                                                   :discard-label-p t)
                                 '(:event "pred." :dataset "test"))))
    (log-mat-room)
    (log-msg "---------------------------------------------------~%")))
