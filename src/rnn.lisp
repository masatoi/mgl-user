;;; -*- coding:utf-8; mode:lisp; -*-

(in-package :tsukuyomi)

(defun get-ave-rate (currency-code session)
  (multiple-value-bind (time rate-list)
      (get-rate session)
    (declare (ignore time))
    (let ((rate-pair (nth currency-code rate-list)))
      (/ (+ (car rate-pair) (cadr rate-pair)) 2))))

(defun make-difference-list (currency-code sd-session &key (n nil))
  (let ((difference-list nil))
    (do ((previous-bid (get-ave-rate currency-code sd-session))
         (current-bid  (get-ave-rate currency-code sd-session))
         (i 0 (1+ i)))
        ((if n (= i n)))
      (push (- current-bid previous-bid) difference-list)
      (setf previous-bid current-bid
            current-bid (get-ave-rate currency-code sd-session)))
    (nreverse difference-list)))

(defparameter sd-session (make-sdat-session nil "/home/wiz/click-data/"))
(initialize-sdat-session! sd-session :day-of-start "2008-03-03")

(defparameter *difference-list* (make-difference-list USDJPY sd-session :N (* 1440 260)))

(defun difference-list->accumulate-list (diff-list)
  (nlet iter ((diff-list diff-list)
              (sum 0)
              (product nil))
    (if (null diff-list)
      (nreverse product)
      (let ((new-sum (+ sum (car diff-list))))
        (iter (cdr diff-list) new-sum (cons new-sum product))))))

(defparameter *difference-vector* (coerce *difference-list* 'vector))

(in-package :mgl-user)

(defun make-fx-sampler (max-n-samples seq-len difference-vector)
  (make-instance 'function-sampler
     :max-n-samples max-n-samples
     :generator (lambda ()
                  (let ((start (random (- (length difference-vector) seq-len))))
                    ;; (format t "start: ~A~%" start) ; debug
                    (list (subseq difference-vector start (+ start seq-len))
                          (aref difference-vector (+ start seq-len)))))))

;; (defparameter sampler (make-fx-sampler 30000 60 tkym::*difference-vector*))

;; (defun read-ucr-dataset (path)
;;   (let* ((data (fare-csv:read-csv-file path))
;;          (datavec (make-array (length data))))
;;     (loop for i from 0 to (1- (length data))
;;           for datum in data
;;           do
;;        (let ((a (make-array (1- (length datum))
;;                             :initial-contents (mapcar #'parse-number:parse-number (cdr datum))))
;;              (label (parse-integer (car datum))))
;;          (setf (aref datavec i)
;;                (make-datum :id (1+ i)
;;                            :label label
;;                            :array (array-to-mat a)))))
;;     datavec))


;; (defparameter train-50word (read-ucr-dataset "/home/wiz/datasets/UCR_TS_Archive_2015/50words/50words_TRAIN"))
;; (defparameter test-50word (read-ucr-dataset "/home/wiz/datasets/UCR_TS_Archive_2015/50words/50words_TEST"))

(defclass fx-fnn (fnn) ())

(defparameter *fx-fnn*
  (let ((n-hiddens 10))
    (build-rnn ()
      (build-fnn (:class 'fx-fnn)
        (input (->input :size 1))
        (h1 (->lstm input :name 'h1 :size n-hiddens))
        (h1-activation (->activation h1 :size 1))
        (prediction (->loss (->squared-difference (activations-output h1-activation)
                                                  (->input :name 'targets :size 1))
                            :name 'prediction))))))

(defmethod set-input (instances (fnn fx-fnn))
  (let ((input-nodes  (nodes (find-clump 'input fnn)))
        (output-nodes (nodes (find-clump 'targets fnn))))
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
