;;; -*- coding:utf-8; mode:lisp; -*-

(in-package :mgl-user)

;;; Methods for RBM
(defmethod set-input (samples (rbm rbm))
  (let ((inputs (find 'inputs (visible-chunks rbm) :key #'name)))
    (when inputs
      (clamp-data samples (nodes inputs)))))

;;; Functions for DBN
(defun init-dbn (dbn &key stddev (start-level 0))
  (loop for i upfrom start-level
        for rbm in (subseq (rbms dbn) start-level) do
          (flet ((this (x)
                   (if (listp x)
                     (elt x i)
                     x)))
            (do-clouds (cloud rbm)
              (if (conditioning-cloud-p cloud)
                (fill! 0 (weights cloud))
                (progn
                  (log-msg "init: ~A ~A~%" cloud (this stddev))
                  (gaussian-random! (weights cloud)
                                    :stddev (this stddev))))))))

(defun train-dbn (dbn training
                  &key n-epochs n-gibbs learning-rate decay visible-sampling (start-level 0))
  (loop
    for rbm in (subseq (rbms dbn) start-level)
    for i upfrom start-level do
      (log-msg "Starting to train level ~S RBM in DBN.~%" i)
      (setf (n-rbms dbn) (1+ i))
      (flet ((this (x) (if (and x (listp x)) (elt x i) x)))
        (let* ((optimizer
                (make-instance 'segmented-gd-optimizer
                   :segmenter
                   (lambda (cloud)
                     (make-instance 'sgd-optimizer
                        :learning-rate (this learning-rate)
                        :weight-decay  (if (conditioning-cloud-p cloud) 0 (this decay))
                        :batch-size    (max-n-stripes dbn)))))
               (gradient-source
                (make-instance 'rbm-cd-learner
                   :rbm rbm
                   :visible-sampling (this visible-sampling)
                   :n-gibbs (this n-gibbs)))
               (dataset (make-sampler training
                                      :n-epochs n-epochs
                                      :sample-visible-p (this visible-sampling))))
          (minimize optimizer gradient-source :dataset dataset)))))

(defun train-dbn-process (dbn training
                          &key (n-epochs 2) (n-gibbs 1) (learning-rate 0.1) (decay 0.0002))
  (with-cuda* ()
    (repeatably ()
      (init-dbn dbn :stddev 0.1)
      (train-dbn dbn training
                 :n-epochs n-epochs :n-gibbs n-gibbs
                 :start-level 0 :learning-rate learning-rate
                 :decay decay :visible-sampling nil)))
  (log-msg "End")
  dbn)

(defun train-dbn-with-monitor (dbn training test
                               &key n-epochs n-gibbs learning-rate
                                 decay visible-sampling (start-level 0))
  (loop
    for rbm in (subseq (rbms dbn) start-level)
    for i upfrom start-level do
      (log-msg "Starting to train level ~S RBM in DBN.~%" i)
      (setf (n-rbms dbn) (1+ i))
      (flet ((this (x) (if (and x (listp x)) (elt x i) x)))
        (let* ((optimizer
                (monitor-optimization-periodically
                 (make-instance 'segmented-gd-optimizer-with-data
                    :training training
                    :test test
                    :segmenter
                    (lambda (cloud)
                      (make-instance 'sgd-optimizer
                         :learning-rate (this learning-rate)
                         :weight-decay  (if (conditioning-cloud-p cloud) 0 (this decay))
                         :batch-size    (max-n-stripes dbn))))
                 '((:fn log-rbm-test-error :period log-test-period)
                   (:fn reset-optimization-monitors
                    :period log-training-period
                    :last-eval 0))))
               (gradient-source
                (make-instance 'rbm-cd-learner
                   :rbm rbm
                   :visible-sampling (this visible-sampling)
                   :n-gibbs (this n-gibbs)))
               (dataset (make-sampler training
                                      :n-epochs n-epochs
                                      :sample-visible-p (this visible-sampling))))
          (minimize optimizer gradient-source :dataset dataset)))))

(defun train-dbn-process-with-monitor (dbn training test
                                       &key (n-epochs 2) (n-gibbs 1)
                                         (learning-rate 0.1) (decay 0.0002))
  (with-cuda* ()
    (repeatably ()
      (init-dbn dbn :stddev 0.1)
      (train-dbn-with-monitor dbn training test
                              :n-epochs n-epochs :n-gibbs n-gibbs
                              :start-level 0 :learning-rate learning-rate
                              :decay decay :visible-sampling nil)))
  (log-msg "End")
  dbn)
