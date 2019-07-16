#lang racket
(provide (all-defined-out))

(define (sequence low high stride)
  (if (> low high)
      null
      (cons low (sequence (+ low stride) high)))
  )

(define (string-append-map xs suffix)
  (map (lambda (str) (string-append str suffix)) xs)
  )

(define (list-nth-mod xs n)
  (if (< n 0)
      (error "list-nth-mod: negative number")
      (if (null? xs)
          (error "list-nth-mod: empty list")
          (car (list-tail xs (- (remainder n (length xs)) 1))))
      )
  )

(define (stream-for-n-steps s n)
  (if (= n 0)
      null
      (let ([pr (s)])
        (cons (car pr) (stream-for-n-steps (cdr pr) (- n 1)))
        )
      )
  )

(define funny-number-stream
  (letrec ([f (lambda (x) (if (= 0 (remainder x 5))
                            (cons (- 0 x) (lambda () (f (+ 1 x))))
                            (cons x (lambda () (f (+ 1 x))))
                            ))]
              )
  (lambda () (f 1)))
  )

(define dan-then-dog
  (letrec ([f (lambda (x)
                (if (string=? x "dan.jpg")
                    (cons x (lambda () (f ("dog.jpg"))))
                    (cons x (lambda () (f ("dan.jpg"))))
                    )
                )
              ]
           )
    (lambda () (f "dan.jpg"))))

(define (stream-add-zero s)
  (let ([pr (s)])
    (lambda () (cons (cons 0 (car pr)) (stream-add-zero (cdr pr))))
    )
  )
