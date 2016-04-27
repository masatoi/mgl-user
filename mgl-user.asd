#|
  This file is a part of mgl-user project.
|#

(in-package :cl-user)
(defpackage mgl-user-asd
  (:use :cl :asdf))
(in-package :mgl-user-asd)

(defsystem mgl-user
  :version "0.1"
  :author ""
  :license ""
  :depends-on (:mgl :mgl-example)
  :components ((:module "src"
                :components
                ((:file "package")
                 (:file "datum" :depends-on ("package"))
                 (:file "load-mnist" :depends-on ("package"))
                 (:file "logging" :depends-on ("package" "datum"))
                 (:file "dbn" :depends-on ("package" "datum" "load-mnist" "logging"))
                 (:file "fnn" :depends-on ("package" "datum" "load-mnist" "logging")))))
  :description ""
  :long-description
  #.(with-open-file (stream (merge-pathnames
                             #p"README.org"
                             (or *load-pathname* *compile-file-pathname*))
                            :if-does-not-exist nil
                            :direction :input)
      (when stream
        (let ((seq (make-array (file-length stream)
                               :element-type 'character
                               :fill-pointer t)))
          (setf (fill-pointer seq) (read-sequence seq stream))
          seq)))
  :in-order-to ((test-op (test-op mgl-user-test))))
