;;; -*- coding:utf-8; mode:lisp; -*-

(defpackage :mgl-user
  (:use :common-lisp :mgl :mgl-example-mnist)
  (:export :make-regression-datum
           :regression-dataset-normalize!
           :copy-regression-dataset
           :regression-datum-id
           :log-regression-cost
           :train-regression-fnn-with-monitor
           :train-regression-fnn-process-with-monitor
           :activations-output
           :find-last-activation
           :predict-regression-datum
           :regression-fnn
           :targets
   ))
