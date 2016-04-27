;;; -*- coding:utf-8; mode:lisp; -*-

(in-package :mgl-user)

;;;; Reading the files

(defun read-low-endian-ub32 (stream)
  (+ (ash (read-byte stream) 24)
     (ash (read-byte stream) 16)
     (ash (read-byte stream) 8)
     (read-byte stream)))

(defun read-magic-or-lose (stream magic)
  (unless (eql magic (read-low-endian-ub32 stream))
    (error "Bad magic.")))

(defun read-image-labels (stream)
  (read-magic-or-lose stream 2049)
  (let ((n (read-low-endian-ub32 stream)))
    (loop repeat n collect (read-byte stream))))

(defun read-image-array (stream)
  (let ((a (make-array (* 28 28))))
    (loop for i below (* 28 28) do
      (let ((pixel (/ (read-byte stream) 255)))
        (unless (<= 0 pixel 1)
          (error "~S" pixel))
        (setf (aref a i) pixel)))
    (array-to-mat a)))

(defun print-image-array (a)
  (with-facets ((a (a 'backing-array :direction :input)))
    (dotimes (y 28)
      (dotimes (x 28)
        (princ (if (< 0.5 (aref a (+ (* y 28) x))) #\O #\.)))
      (terpri))))

(defun read-image-arrays (stream)
  (read-magic-or-lose stream 2051)
  (let ((n (read-low-endian-ub32 stream)))
    (read-magic-or-lose stream 28)
    (read-magic-or-lose stream 28)
    (coerce (loop repeat n collect (read-image-array stream))
            'vector)))

(defparameter *mnist-data-dir*
  (merge-pathnames "mnist-data/" "/home/wiz/mgl/example/")
  "Set this to the directory where the uncompressed mnist files reside.")

(defparameter *mnist-save-dir*
  (merge-pathnames "mnist-save/" "/home/wiz/mgl/example/")
  "Set this to the directory where the trained models are saved.")

(defun load-training (&optional (mnist-data-dir *mnist-data-dir*))
  (log-msg "Loading training images~%")
  (prog1
      (let ((id 0))
        (map 'vector
             (lambda (label array)
               (make-datum :id (incf id) :label label :array array))
             (with-open-file (s (merge-pathnames "train-labels-idx1-ubyte"
                                                 mnist-data-dir)
                                :element-type 'unsigned-byte)
               (read-image-labels s))
             (with-open-file (s (merge-pathnames "train-images-idx3-ubyte"
                                                 mnist-data-dir)
                                :element-type 'unsigned-byte)
               (read-image-arrays s))))
    (log-msg "Loading training images done~%")))

(defun load-test (&optional (mnist-data-dir *mnist-data-dir*))
  (log-msg "Loading test images~%")
  (prog1
      (let ((id 0))
        (map 'vector
             (lambda (label array)
               (make-datum :id (decf id) :label label :array array))
             (with-open-file (s (merge-pathnames "t10k-labels-idx1-ubyte"
                                                 mnist-data-dir)
                                :element-type 'unsigned-byte)
               (read-image-labels s))
             (with-open-file (s (merge-pathnames "t10k-images-idx3-ubyte"
                                                 mnist-data-dir)
                                :element-type 'unsigned-byte)
               (read-image-arrays s))))
    (log-msg "Loading test images done~%")))

(defvar *training-data*)
(defvar *test-data*)

(defun training-data ()
  (unless (boundp '*training-data*)
    (setq *training-data* (load-training)))
  *training-data*)

(defun test-data ()
  (unless (boundp '*test-data*)
    (setq *test-data* (load-test)))
  *test-data*)

(defun find-image-by-id (id)
  (find id (if (plusp id)
               (training-data)
               (test-data))
        :key #'datum-id))
