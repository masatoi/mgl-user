;;; -*- coding:utf-8; mode:lisp; -*-

(in-package :mgl-user)

(defstruct datum
  (id nil :type fixnum)
  (label nil)
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

(defun clamp-data (samples mat)
  (assert (= (length samples) (mat-dimension mat 0)))
  (map-concat #'copy! samples mat :key #'sample-datum-array))

;; for clamp-chunk-labels of DBN/DBM
(defun clamp-labels (samples mat)
  (assert (= (length samples) (mat-dimension mat 0)))
  (fill! 0 mat)
  (let ((n-columns (mat-dimension mat 1))
        (displacement (mat-displacement mat))
        (one (coerce-to-ctype 1 :ctype (mat-ctype mat))))
    (with-facets ((a (mat 'backing-array :direction :io)))
      (loop for sample in samples
            for row upfrom 0
            do (destructuring-bind (image &key discard-label-p sample-visible-p)
                   sample
                 (declare (ignore sample-visible-p))
                 (unless discard-label-p
                   (setf (aref a (+ displacement
                                    (* row n-columns)
                                    (image-label image)))
                         one)))))))

;;; Grobal Contrast Normalization

;; AXPY! ALPHA X Y &KEY (N (MAT-SIZE X)) (INCX 1) (INCY 1)

(defun dataset-average (dataset)
  (let* ((first-datum (datum-array (aref dataset 0)))
         (datum-dim (mat-dimension first-datum 0))
         (result (make-mat datum-dim)))
    (loop for datum across dataset do
      (axpy! (/ 1.0 (length dataset)) (datum-array datum) result))
    result))

;; input of wiz:splot-matrix
(defun mat-to-28x28-array (mat)
  (let ((mat2a (mat-to-array mat))
        (a (make-array '(28 28))))
    (loop for i from 0 to 27 do
      (loop for j from 0 to 27 do
        (setf (aref a i j) (aref mat2a (+ (* i 28) j)))))
    a))

;; (loop for i from 0 to 9 do
;;   (wiz:splot-matrix (mat-to-28x28-array
;;                      (dataset-average
;;                       (remove-if-not
;;                        (lambda (datum) (= (datum-label datum) i))
;;                        *training-data*)))))

(defun dataset-variance (dataset average)
  (let* ((first-datum (datum-array (aref dataset 0)))
         (datum-dim (mat-dimension first-datum 0))
         (result (make-mat datum-dim))
         (diff (make-mat datum-dim)))
    (loop for datum across dataset do
      (copy! average diff)
      (axpy! -1.0 (datum-array datum) diff)
      (.square! diff)
      (axpy! (/ 1.0 (length dataset)) diff result))
    result))

(defun copy-dataset (dataset)
  (let ((new-dataset (map 'vector (lambda (datum) (copy-datum datum)) dataset)))
    (loop for new-datum across new-dataset
          for datum across dataset do
      (setf (datum-array new-datum) (copy-mat (datum-array datum))))
    new-dataset))

(defun dataset-normalize! (dataset &key test-dataset (noise-degree 1.0))
  (let* ((first-datum (datum-array (aref dataset 0)))
         (datum-dim (mat-dimension first-datum 0))
         (average (dataset-average dataset))
         (variance (dataset-variance dataset average))
         (noise (make-mat datum-dim :initial-element noise-degree)))
    (axpy! 1.0 noise variance)
    (.sqrt! variance)
    (.inv! variance)
    (loop for datum across dataset do
      (axpy! -1.0 average (datum-array datum))
      (geem! 1.0 variance (datum-array datum) 0.0 (datum-array datum)))
    (if test-dataset
      (loop for datum across test-dataset do
        (axpy! -1.0 average (datum-array datum))
        (geem! 1.0 variance (datum-array datum) 0.0 (datum-array datum))))
    'done))
