#|
  This file is a part of mgl-user project.
|#

(in-package :cl-user)
(defpackage mgl-user-test-asd
  (:use :cl :asdf))
(in-package :mgl-user-test-asd)

(defsystem mgl-user-test
  :author ""
  :license ""
  :depends-on (:mgl-user
               :prove)
  :components ((:module "t"
                :components
                ((:test-file "mgl-user"))))
  :description "Test system for mgl-user"

  :defsystem-depends-on (:prove-asdf)
  :perform (test-op :after (op c)
                    (funcall (intern #.(string :run-test-system) :prove-asdf) c)
                    (asdf:clear-system c)))
