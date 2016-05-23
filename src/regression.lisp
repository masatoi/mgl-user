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

;;; Physicochemical Properties of Protein Tertiary Structure Data Set
;;; http://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure
(defun read-casp-dataset (path)
  (let* ((data (fare-csv:read-csv-file path))
         (datavec (make-array (length data))))
    (loop for i from 0 to (1- (length data))
          for datum in data
          do
       (let ((a (make-array (1- (length datum))
                            :initial-contents (mapcar #'parse-number:parse-number (cdr datum))))
             (target (make-array 1 :initial-contents (list (* (parse-number:parse-number (car datum)) 1.0)))))
         (setf (aref datavec i)
               (make-regression-datum :id (1+ i)
                                      :target (array-to-mat target)
                                      :array (array-to-mat a)))))
    datavec))

(defparameter casp-dataset (read-casp-dataset "/home/wiz/datasets/CASP.csv"))

(defclass regression-fnn (fnn) ())

(defparameter fnn-regression
  (build-fnn (:class 'regression-fnn :max-n-stripes 100)
    ;; Input Layer
    (inputs (->input :size 9))
    (f1-activations (->activation inputs :name 'f1 :size 256))
    (f1 (->relu f1-activations))
    (f2-activations (->activation f1 :name 'f2 :size 256))
    (f2 (->relu f2-activations))
    ;; Output Lump: ->squared-difference
    (prediction (->loss (->squared-difference (->activation f2 :name 'prediction :size 1)
                                              (->input :name 'targets :size 1))
                        :name 'squared-error))))

(defmethod set-input (samples (bpn regression-fnn))
  (let* ((inputs (find-clump 'inputs bpn))
         (targets (find-clump 'targets bpn)))
    (clamp-data samples (nodes inputs))
    (setf (target prediction) (label-target-list samples))))

(defmethod set-input (instances (fnn my-rnn))
  (let ((input-nodes (nodes (find-clump 'input fnn))))
    (setf (target (find-clump 'prediction fnn))
          (loop for stripe upfrom 0
                for instance in instances
                collect
                ;; Sequences in the batch are not of equal length. The
                ;; RNN sends a NIL our way if a sequence has run out.
                (when instance
                  (destructuring-bind (input target) instance
                    (setf (mref input-nodes stripe 0) input)
                    target))))))

;; Monitor functions have to display not accuracy but error for test dataset.
;; This dataset doesn't have test dataset, therefore have to do cross validation. => Resampling

