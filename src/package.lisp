;;; -*- coding:utf-8; mode:lisp; -*-

(defpackage :mgl-user
  (:use :common-lisp :mgl :mgl-example-mnist)
  (:export :make-regression-datum
           :regression-dataset-normalize!
           :log-regression-cost
           :train-regression-fnn-with-monitor
           :train-regression-fnn-process-with-monitor
           :activations-output
           :find-last-activation
           :predict-regression-datum
   ))
