;;; -*- coding:utf-8; mode:lisp; -*-

(in-package :mgl-user)

;;; Rastrigin function

(defun rastrigin (x-list)
  (let ((n (length x-list)))
    (+ (* 10 n)
       (loop for xi in x-list summing
	 (- (* xi xi) (* 10 (cos (* 2 pi xi))))))))

(defparameter *rastrigin-dataset*
  (let* ((n 10000)
         (arr (make-array n)))
    (loop for i from 0 to (1- n) do
      (let ((x (- (random 10.24) 5.12))
            (y (- (random 10.24) 5.12)))
        (setf (aref arr i) (make-regression-datum
                            :id (1+ i)
                            :target (make-mat 1 :initial-element (rastrigin (list x y)))
                            :array  (make-mat 2 :initial-contents (list x y))))))
    arr))

(defparameter *rastrigin-test*
  (let* ((n (* 64 64)) ; separate to 64x64 cells (by 0.16)
         (arr (make-array n)))
    (loop for i from 0 to 63 do
      (loop for j from 0 to 63 do
        (let ((x (- (* i 0.16) 5.04))
              (y (- (* j 0.16) 5.04)))
          (setf (aref arr (+ (* i 64) j))
                (make-regression-datum
                 :id (1+ (+ (* i 64) j))
                 :target (make-mat 1 :initial-element (rastrigin (list x y)))
                 :array  (make-mat 2 :initial-contents (list x y)))))))
    arr))

(defparameter *rastrigin-dataset-normal* (copy-regression-dataset *rastrigin-dataset*))
(defparameter *rastrigin-test-normal* (copy-regression-dataset *rastrigin-test*))
(regression-dataset-normalize! *rastrigin-dataset-normal*
                               :test-dataset *rastrigin-test-normal*
                               :noise-degree 0.0)

(defparameter fnn-rastrigin
  (build-fnn (:class 'regression-fnn :max-n-stripes 100)
    ;; Input Layer
    (inputs (->input :size 2))
    (f1-activations (->activation inputs :name 'f1 :size 256))
    (f1 (->relu f1-activations))
    (f2-activations (->activation f1 :name 'f2 :size 256))
    (f2 (->relu f2-activations))
    (prediction-activations (->activation f2 :name 'prediction :size 1))
    ;; Output Lump: ->squared-difference
    (prediction (->loss (->squared-difference (activations-output prediction-activations)
                                              (->input :name 'targets :size 1))
                        :name 'prediction))))

;; (train-bpn-gd-process fnn-rastrigin *rastrigin-dataset-normal*
;;                                 :l2-upper-bound 1.0
;;                                 :input-weight-penalty 0.000001
;;                                 :n-epochs 500)

;; (train-regression-fnn-process-with-monitor
;;  fnn-rastrigin *rastrigin-dataset-normal*
;;  :test *rastrigin-test-normal* :n-epochs 10)

(defparameter fnn-maxout-dropout-regression
  (let ((group-size 5))
    (build-fnn (:class 'regression-fnn :max-n-stripes 100)
      (inputs (->input :size 2 :dropout 0.2))
      (f1-activations (->activation inputs :name 'f1 :size 1200))
      (f1* (->max f1-activations :group-size group-size))
      (f1 (->dropout f1* :dropout 0.5))
      (f2-activations (->activation f1 :name 'f2 :size 1200))
      (f2* (->max f2-activations :group-size group-size))
      (f2 (->dropout f2* :dropout 0.5))
      (f3-activations (->activation f2 :name 'f3 :size 1200))
      (f3* (->max f3-activations :group-size group-size))
      (f3 (->dropout f3* :dropout 0.5))
      (prediction-activations (->activation f3 :name 'prediction :size 1))
      (prediction (->loss (->squared-difference (activations-output prediction-activations)
                                                (->input :name 'targets :size 1))
                          :name 'prediction)))))

(defparameter fnn-regression
  (build-fnn (:class 'regression-fnn :max-n-stripes 100)
    ;; Input Layer
    (inputs (->input :size 2))
    (f1-activations (->activation inputs :name 'f1 :size 1200))
    (f1 (->relu f1-activations))
    (f2-activations (->activation f1 :name 'f2 :size 1200))
    (f2 (->relu f2-activations))
    (f3-activations (->activation f2 :name 'f3 :size 1200))
    (f3 (->relu f3-activations))
    (prediction-activations (->activation f3 :name 'prediction :size 1))
    ;; Output Lump: ->squared-difference
    (prediction (->loss (->squared-difference (activations-output prediction-activations)
                                              (->input :name 'targets :size 1))
                        :name 'prediction))))

(train-regression-fnn-process-with-monitor
 fnn-regression *rastrigin-dataset-normal*
 :test *rastrigin-test-normal* :n-epochs 500)

;;; Prediction

(defparameter test-map
  (map 'vector (lambda (datum)
                 (mref (regression-datum-target datum) 0))
       *rastrigin-test-normal*))

(defparameter prediction-map
  (map 'vector (lambda (datum)
                 (mref (predict-regression-datum fnn-rastrigin datum) 0))
       *rastrigin-test-normal*))

(defparameter prediction-map
  (map 'vector (lambda (datum)
                 (mref (predict-regression-datum fnn-regression datum) 0))
       *rastrigin-test-normal*))

;; (wiz:splot-matrix (array-to-64x64-array test-map) :output "/home/wiz/Dropbox/tmp/test-map612.png")
;; (wiz:splot-matrix (array-to-64x64-array prediction-map) :output "/home/wiz/Dropbox/tmp/prediction-map612-3.png")

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

;; (defparameter *casp-dataset* (read-casp-dataset "/home/wiz/datasets/CASP.csv"))
;; (defparameter *casp-dataset-normal* (copy-regression-dataset *casp-dataset*))
;; (regression-dataset-normalize! *casp-dataset-normal* :noise-degree 0.0)
;; (train-fnn-process fnn-regression *casp-dataset-normal* :n-epochs 10)

;;; Define model

(defparameter fnn-regression
  (build-fnn (:class 'regression-fnn :max-n-stripes 100)
    ;; Input Layer
    (inputs (->input :size 9))
    (f1-activations (->activation inputs :name 'f1 :size 256))
    (f1 (->relu f1-activations))
    (f2-activations (->activation f1 :name 'f2 :size 256))
    (f2 (->relu f2-activations))
    (prediction-activations (->activation f2 :name 'prediction :size 1))
    ;; Output Lump: ->squared-difference
    (prediction (->loss (->squared-difference (activations-output prediction-activations)
                                              (->input :name 'targets :size 1))
                        :name 'prediction))))

(defparameter fnn-regression
  (build-fnn (:class 'regression-fnn :max-n-stripes 100)
    ;; Input Layer
    (inputs (->input :size 9))
    (f1-activations (->activation inputs :name 'f1 :size 1200))
    (f1 (->relu f1-activations))
    (f2-activations (->activation f1 :name 'f2 :size 1200))
    (f2 (->relu f2-activations))
    (f3-activations (->activation f2 :name 'f3 :size 1200))
    (f3 (->relu f3-activations))
    (prediction-activations (->activation f3 :name 'prediction :size 1))
    ;; Output Lump: ->squared-difference
    (prediction (->loss (->squared-difference (activations-output prediction-activations)
                                              (->input :name 'targets :size 1))
                        :name 'prediction))))

(defparameter fnn-relu-dropout-regression
  (build-fnn (:class 'regression-fnn :max-n-stripes 100)
    (inputs (->input :size 9 :dropout 0.2))
    (f1-activations (->activation inputs :name 'f1 :size 1200))
    (f1* (->relu f1-activations))
    (f1 (->dropout f1*))
    (f2-activations (->activation f1 :name 'f2 :size 1200))
    (f2* (->relu f2-activations))
    (f2 (->dropout f2*))
    (f3-activations (->activation f2 :name 'f3 :size 1200))
    (f3* (->relu f3-activations))
    (f3 (->dropout f3*))
    (prediction-activations (->activation f3 :name 'prediction :size 1))
    (prediction (->loss (->squared-difference (activations-output prediction-activations)
                                              (->input :name 'targets :size 1))
                        :name 'prediction))))

(defparameter fnn-maxout-dropout-regression
  (let ((group-size 5))
    (build-fnn (:class 'regression-fnn :max-n-stripes 100)
      (inputs (->input :size 9 :dropout 0.2))
      (f1-activations (->activation inputs :name 'f1 :size 1200))
      (f1* (->max f1-activations :group-size group-size))
      (f1 (->dropout f1* :dropout 0.5))
      (f2-activations (->activation f1 :name 'f2 :size 1200))
      (f2* (->max f2-activations :group-size group-size))
      (f2 (->dropout f2* :dropout 0.5))
      (f3-activations (->activation f2 :name 'f3 :size 1200))
      (f3* (->max f3-activations :group-size group-size))
      (f3 (->dropout f3* :dropout 0.5))
      (prediction-activations (->activation f3 :name 'prediction :size 1))
      (prediction (->loss (->squared-difference (activations-output prediction-activations)
                                                (->input :name 'targets :size 1))
                          :name 'prediction)))))

;; (train-regression-fnn-process-with-monitor fnn-regression *casp-dataset-normal* :n-epochs 10)
