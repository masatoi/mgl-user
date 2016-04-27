;;; -*- coding:utf-8; mode:lisp; -*-

(in-package :mgl-user)

(defstruct datum
  (id nil :type fixnum)
  (label nil :type (integer 0 10))
  (array nil :type mat))

;;;; Sampling, clamping, utilities

(defun make-sampler (data &key (n-epochs 1)
                     (max-n (* n-epochs (length data)))
                     discard-label-p sample-visible-p)
  (make-instance 'function-sampler
                 :max-n-samples max-n
                 :generator (let ((g (make-random-generator data)))
                              (lambda ()
                                (list (funcall g)
                                      :discard-label-p discard-label-p
                                      :sample-visible-p sample-visible-p)))))

(defun make-tiny-sampler (data &key discard-label-p)
  (make-sampler (subseq data 0 1000) :discard-label-p discard-label-p))

(defun sample-datum-array (sample)
  (datum-array (first sample)))

(defun clamp-data (samples mat)
  (assert (= (length samples) (mat-dimension mat 0)))
  (map-concat #'copy! samples mat :key #'sample-datum-array))

(defun clamp-labels (samples mat)
  (assert (= (length samples) (mat-dimension mat 0)))
  (fill! 0 mat)
  (let ((n-columns (mat-dimension mat 1))
        (displacement (mat-displacement mat))
        (one (coerce-to-ctype 1 :ctype (mat-ctype mat))))
    (with-facets ((a (mat 'backing-array :direction :io)))
      (loop for sample in samples
            for row upfrom 0
            do (destructuring-bind (datum &key discard-label-p sample-visible-p)
                   sample
                 (declare (ignore sample-visible-p))
                 (unless discard-label-p
                   (setf (aref a (+ displacement
                                    (* row n-columns)
                                    (datum-label datum)))
                         one)))))))
