;;; -*- coding:utf-8; mode:lisp; -*-

(in-package :mgl-user)

(ql:quickload :fare-csv)
(ql:quickload :parse-number)

(defun read-ucr-dataset (path)
  (let* ((data (fare-csv:read-csv-file path))
         (datavec (make-array (length data))))
    (loop for i from 0 to (1- (length data))
          for datum in data
          do
       (let ((a (make-array (1- (length datum))
                            :initial-contents (mapcar #'parse-number:parse-number (cdr datum))))
             (label (parse-integer (car datum))))
         (setf (aref datavec i)
               (make-datum :id (1+ i)
                           :label label
                           :array (array-to-mat a)))))
    datavec))


(defparameter train-50word (read-ucr-dataset "/home/wiz/datasets/UCR_TS_Archive_2015/50words/50words_TRAIN"))
(defparameter test-50word (read-ucr-dataset "/home/wiz/datasets/UCR_TS_Archive_2015/50words/50words_TEST"))

(defclass my-rnn (fnn)
  ())

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

(defparameter rnn1
  (build-rnn ()
    (build-fnn (:class 'my-rnn :max-n-stripes 100)
      (input (->input :size 1))
      (h (->lstm input :name 'h :size 10))
      (prediction (->softmax-xe-loss (->activation h :name 'prediction :size 50))))))
