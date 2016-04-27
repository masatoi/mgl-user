;;; -*- coding:utf-8; mode:lisp; -*-

;;;; Code for the MNIST handwritten digit recognition challange.
;;;;
;;;; References:
;;;;
;;;; For the DBN-to-BPN approach:
;;;;
;;;;   To Recognize Shapes, First Learn to Generate Images,
;;;;   http://www.cs.toronto.edu/~hinton/absps/montrealTR.pdf
;;;;
;;;;   http://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html
;;;;
;;;; and for the DBN-to-DBM-to-BPN one:
;;;;
;;;;   "Deep Boltzmann Machines",
;;;;   http://www.cs.toronto.edu/~hinton/absps/dbm.pdf
;;;;
;;;; For dropout:
;;;;
;;;;   "Improving neural networks by preventing co-adaptation of
;;;;   feature detectors"
;;;;   http://arxiv.org/pdf/1207.0580.pdf
;;;;
;;;; Maxout:
;;;;
;;;;  "Maxout Networks"
;;;;  http://arxiv.org/abs/1302.4389
;;;;
;;;; Download the four files from http://yann.lecun.com/exdb/mnist and
;;;; gunzip them. Set *MNIST-DATA-DIR* to point to their directory.

(in-package :mgl-user)

;;; Load training/test data
(defparameter *mnist-data-dir* "/home/wiz/mgl/example/mnist-data/")
(defparameter *mnist-save-dir* "/home/wiz/mgl/example/mnist-save/")

;; Set to *training-data* and  *test-data*
(progn (training-data)
       (test-data)
       'done)

;;; FFNN ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defparameter fnn1
  (build-fnn (:class 'fnn :max-n-stripes 100)
    ;; Input Layer
    (inputs (->input :size 784))
    ;; Layer1 500units, ReLU
    (f1-activations (->activation inputs :name 'f1 :size 500))
    (f1 (->relu f1-activations)) ;  ->SIGMOID, ->DROPOUT, ->RELU, ->TANH, etc...
    ;; Layer2 500 units, ReLU
    (f2-activations (->activation f1 :name 'f2 :size 500))
    (f2 (->relu f2-activations))
    ;; Layer3 2000 units, ReLU
    (f3-activations (->activation f2 :name 'f3 :size 2000))
    (f3 (->relu f3-activations))
    ;; Output Layer: softmax layer
    (prediction (->softmax-xe-loss (->activation f3 :name 'prediction :size 10)
                                   :name 'prediction))))

(defparameter fnn1
  (build-fnn (:class 'fnn :max-n-stripes 100)
    ;; Input Layer
    (inputs (->input :size 784))
    ;; Layer1 500units, ReLU
    (f1-activations (->activation inputs :name 'f1 :size 500))
    (f1 (->relu f1-activations)) ;  ->SIGMOID, ->DROPOUT, ->RELU, ->TANH, etc...
    ;; Layer2 500 units, ReLU
    (f2-activations (->activation f1 :name 'f2 :size 500))
    (f2 (->relu f2-activations))
    ;; Layer3 2000 units, ReLU
    (f3-activations (->activation f2 :name 'f3 :size 2000))
    (f3 (->relu f3-activations))
    ;; Output Layer: softmax layer
    (prediction (->softmax-xe-loss (->activation f3 :name 'prediction :size 10)
                                   :name 'prediction))))

(defparameter fnn-sigmoid
  (build-fnn (:class 'fnn :max-n-stripes 100)
    ;; Input Layer
    (inputs (->input :size 784))
    ;; Layer1 500units, ReLU
    (f1-activations (->activation inputs :name 'f1 :size 500))
    (f1 (->sigmoid f1-activations)) ;  ->SIGMOID, ->DROPOUT, ->RELU, ->TANH, etc...
    ;; Layer2 500 units, ReLU
    (f2-activations (->activation f1 :name 'f2 :size 500))
    (f2 (->sigmoid f2-activations))
    ;; Layer3 2000 units, ReLU
    (f3-activations (->activation f2 :name 'f3 :size 2000))
    (f3 (->sigmoid f3-activations))
    ;; Output Layer: softmax layer
    (prediction (->softmax-xe-loss (->activation f3 :name 'prediction :size 10)
                                   :name 'prediction))))

(defparameter fnn-sigmoid2
  (build-fnn (:class 'fnn :max-n-stripes 100)
    ;; Input Layer
    (inputs (->input :size 784))
    ;; Layer1 500units, ReLU
    (f1-activations (->activation inputs :name 'f1 :size 500))
    (f1 (->sigmoid f1-activations)) ;  ->SIGMOID, ->DROPOUT, ->RELU, ->TANH, etc...
    ;; Output Layer: softmax layer
    (prediction (->softmax-xe-loss (->activation f1 :name 'prediction :size 10)
                                   :name 'prediction))))

(defparameter fnn2
  (build-fnn (:class 'fnn :max-n-stripes 100)
    ;; Input Layer
    (inputs (->input :size 784))
    ;; Layer1 500units, ReLU
    (f1-activations (->activation inputs :name 'f1 :size 1200))
    (f1 (->relu f1-activations))
    ;; Layer2 500 units, ReLU
    (f2-activations (->activation f1 :name 'f2 :size 1200))
    (f2 (->relu f2-activations))
    ;; Layer3 2000 units, ReLU
    (f3-activations (->activation f2 :name 'f3 :size 1200))
    (f3 (->relu f3-activations))
    ;; Output Layer: softmax layer
    (prediction (->softmax-xe-loss (->activation f3 :name 'prediction :size 10)
                                   :name 'prediction))))

(defparameter fnn3
  (build-fnn (:class 'fnn :max-n-stripes 100)
    ;; Input Layer
    (inputs (->input :size 784))
    ;; Layer1 500units, ReLU
    (f1-activations (->activation inputs :name 'f1 :size 500))
    (f1 (->relu f1-activations))
    ;; Layer2 500 units, ReLU
    (f2-activations (->activation f1 :name 'f2 :size 500))
    (f2 (->relu f2-activations))
    ;; Layer3 2000 units, ReLU
    (f3-activations (->activation f2 :name 'f3 :size 500))
    (f3 (->relu f3-activations))
    ;; Layer3 2000 units, ReLU
    (f4-activations (->activation f3 :name 'f4 :size 500))
    (f4 (->relu f4-activations))
    ;; Layer3 2000 units, ReLU
    (f5-activations (->activation f4 :name 'f5 :size 500))
    (f5 (->relu f5-activations))
    ;; Output Layer: softmax layer
    (prediction (->softmax-xe-loss (->activation f5 :name 'prediction :size 10)
                                   :name 'prediction))))

(defparameter fnn-relu-dropout
  (build-fnn (:class 'fnn :max-n-stripes 100)
    (inputs (->input :size 784 :dropout 0.2))
    (f1-activations (->activation inputs :name 'f1 :size 500))
    (f1* (->relu f1-activations))
    (f1 (->dropout f1*))
    (f2-activations (->activation f1 :name 'f2 :size 500))
    (f2* (->relu f2-activations))
    (f2 (->dropout f2*))
    (f3-activations (->activation f2 :name 'f3 :size 2000))
    (f3* (->relu f3-activations))
    (f3 (->dropout f3*))
    (prediction (->softmax-xe-loss (->activation f3 :name 'prediction :size 10)
                                   :name 'prediction))))

(defparameter fnn-maxout-dropout
  (let ((group-size 5))
    (build-fnn (:class 'fnn :max-n-stripes 100)
      (inputs (->input :size 784 :dropout 0.2))
      (f1-activations (->activation inputs :name 'f1 :size 1200))
      (f1* (->max f1-activations :group-size group-size))
      (f1 (->dropout f1*))
      (f2-activations (->activation f1 :name 'f2 :size 1200))
      (f2* (->max f2-activations :group-size group-size))
      (f2 (->dropout f2*))
      (prediction (->softmax-xe-loss (->activation f2 :name 'prediction :size 10)
                                     :name 'prediction)))))

;; MGL-USER> (let ((*cuda-enabled* nil))
;;             (time (train-fnn-process fnn1 *training-data* :n-epochs 10)))
;; 2016-04-22 03:56:05: End
;; Evaluation took:
;;   189.422 seconds of real time
;;   755.168564 seconds of total run time (609.898586 user, 145.269978 system)
;;   [ Run times consist of 0.779 seconds GC time, and 754.390 seconds non-GC time. ]
;;   398.67% CPU
;;   642,546,420,280 processor cycles
;;   1,803,666,912 bytes consed
  
;; #<FNN :STRIPES 100/100 :CLUMPS 9>

;; MGL-USER> (time (train-fnn-process fnn1 *training-data* :n-epochs 10))
;; 2016-04-22 04:00:47: End
;; Evaluation took:
;;   213.962 seconds of real time
;;   214.038342 seconds of total run time (179.886267 user, 34.152075 system)
;;   [ Run times consist of 0.286 seconds GC time, and 213.753 seconds non-GC time. ]
;;   100.04% CPU
;;   725,789,966,866 processor cycles
;;   1,607,094,272 bytes consed

;; MGL-USER> (test-fnn fnn1 *test-data*)
;; (#<CLASSIFICATION-ACCURACY-COUNTER bpn PREDICTION acc.: 98.22% (10000)>
;;  #<CROSS-ENTROPY-COUNTER bpn PREDICTION xent: 7.782d-4 (10000)>)
;; MGL-USER> (test-fnn fnn1 *training-data*)
;; (#<CLASSIFICATION-ACCURACY-COUNTER bpn PREDICTION acc.: 99.74% (60000)>
;;  #<CROSS-ENTROPY-COUNTER bpn PREDICTION xent: 8.069d-5 (60000)>)

;;; Run training

;; by GPU
(train-fnn-process fnn1 *training-data* :n-epochs 10)

;; by CPU
(let ((*cuda-enabled* nil))
  (time (train-fnn-process fnn1 *training-data* :n-epochs 10)))

(let ((*cuda-enabled* nil))
  (time (train-fnn-process-with-monitor
         fnn1
         *training-data*
         *test-data*
         :n-epochs 10)))

;; test
(test-fnn fnn1 *test-data*)
(test-fnn fnn1 *training-data*)

;;; DBN ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Make DBN instance
(defparameter dbn1
  (make-instance 'dbn
     :layers (list (list (make-instance 'constant-chunk :name 'c0)
                         (make-instance 'sigmoid-chunk :name 'inputs
                                        :size (* 28 28)))
                   (list (make-instance 'constant-chunk :name 'c1)
                         (make-instance 'sigmoid-chunk :name 'f1
                                        :size 500))
                   (list (make-instance 'constant-chunk :name 'c2)
                         (make-instance 'sigmoid-chunk :name 'f2
                                        :size 500))
                   (list (make-instance 'constant-chunk :name 'c3)
                         (make-instance 'sigmoid-chunk :name 'f3
                                        :size 2000)))
     :rbm-class 'rbm
     :max-n-stripes 100)) ; batch size

;;; Run training
(train-dbn-process dbn1 *training-data* :n-epochs 50)

(defun train-dbn-process-cpu (dbn training
                              &key (n-epochs 2) (n-gibbs 1) (learning-rate 0.1) (decay 0.0002))
  (init-dbn dbn :stddev 0.1)
  (train-dbn dbn training
             :n-epochs n-epochs :n-gibbs n-gibbs
             :start-level 0 :learning-rate learning-rate
             :decay decay :visible-sampling nil)
  (log-msg "End")
  dbn)

(train-dbn-process-cpu dbn1 *training-data* :n-epochs 50)

(require :sb-sprof)

(sb-sprof:with-profiling (:max-samples 1000
				       :report :flat
				       :loop nil)
  (train-dbn-process dbn1 *training-data* :n-epochs 2))

(sb-sprof:with-profiling (:max-samples 1000
				       :report :flat
				       :loop nil)
  (train-dbn-process-cpu dbn1 *training-data* :n-epochs 2))

(defparameter dbn2
  (make-instance 'dbn
     :layers (list (list (make-instance 'constant-chunk :name 'c0)
                         (make-instance 'sigmoid-chunk :name 'inputs
                                        :size (* 28 28)))
                   (list (make-instance 'constant-chunk :name 'c1)
                         (make-instance 'sigmoid-chunk :name 'f1
                                        :size 1000))
                   (list (make-instance 'constant-chunk :name 'c2)
                         (make-instance 'sigmoid-chunk :name 'f2
                                        :size 1000))
                   (list (make-instance 'constant-chunk :name 'c3)
                         (make-instance 'sigmoid-chunk :name 'f3
                                        :size 2000)))
     :rbm-class 'rbm
     :max-n-stripes 100 ; batch size
     ))

(defparameter dbn3
  (make-instance 'dbn
     :layers (list (list (make-instance 'constant-chunk :name 'c0)
                         (make-instance 'sigmoid-chunk :name 'inputs
                                        :size (* 28 28)))
                   (list (make-instance 'constant-chunk :name 'c1)
                         (make-instance 'sigmoid-chunk :name 'f1
                                        :size 1000))
                   (list (make-instance 'constant-chunk :name 'c2)
                         (make-instance 'sigmoid-chunk :name 'f2
                                        :size 1000))
                   (list (make-instance 'constant-chunk :name 'c3)
                         (make-instance 'sigmoid-chunk :name 'f3
                                        :size 2000)))
     :rbm-class 'rbm
     :max-n-stripes 1000 ; batch size
     ))

(defparameter dbn4
  (make-instance 'dbn
     :layers (list (list (make-instance 'constant-chunk :name 'c0)
                         (make-instance 'sigmoid-chunk :name 'inputs
                                        :size (* 28 28)))
                   (list (make-instance 'constant-chunk :name 'c1)
                         (make-instance 'sigmoid-chunk :name 'f1
                                        :size 4000))
                   (list (make-instance 'constant-chunk :name 'c2)
                         (make-instance 'sigmoid-chunk :name 'f2
                                        :size 4000))
                   (list (make-instance 'constant-chunk :name 'c3)
                         (make-instance 'sigmoid-chunk :name 'f3
                                        :size 2000)))
     :rbm-class 'rbm
     :max-n-stripes 100 ; batch size
     ))
