#lang racket
(provide (all-defined-out))

(define (sequence low high stride)
  (if (> low high)
      null
      (cons low (sequence (+ low stride) high stride)))
  )

(define (string-append-map xs suffix)
  (map (lambda (str) (string-append str suffix)) xs)
  )

(define (list-nth-mod xs n)
  (cond [(< n 0) (error "list-nth-mod: negative number")]
        [(null? xs) (error "list-nth-mod: empty list")]
        [#t (car (list-tail xs (remainder n (length xs))))]
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
                    (cons x (lambda () (f "dog.jpg")))
                    (cons x (lambda () (f "dan.jpg")))
                    )
                )
              ])
    (lambda () (f "dan.jpg"))
    )
  )

(define (stream-add-zero s)
  (let ([pr (s)])
    (lambda () (cons (cons 0 (car pr)) (stream-add-zero (cdr pr))))
    )
  )

(define (cycle-lists xs ys)
  (letrec ([helper (lambda (n)
                     (cons (cons (list-nth-mod xs n) (list-nth-mod ys n)) (lambda () (helper (+ n 1))))
                     )
                   ])
    (lambda () (helper 0))
    )
  )

(define (vector-assoc v vec)
  (letrec ([helper (lambda (n)
                     (if (>= n (vector-length vec))
                         #f
                         (let ([x (vector-ref vec n)])
                           (if (and (pair? x) (equal? (car x) v))
                               x
                               (helper (+ n 1))
                               )
                           )
                         )
                     )])
    (helper 0)
    )
  )

(define (cached-assoc xs n)
  (let* ([cache (make-vector n #f)]
        [i 0]
        [update (lambda (p)
                  (if (>= i (vector-length cache))
                      (set! i 0)
                      (set! i i)
                      )
                  (vector-set! cache i p)
                  (set! i (+ i 1))
                  p
                  )]
        [f (lambda (v) (let ([x (vector-assoc v cache)])
                         (if x
                             x
                             (let ([y (assoc v xs)])
                               (if y
                                   (update y)
                                   #f
                                   )
                               )
                             )
                         )
             )]
        )
    f
    )
  )

(define-syntax while-less
  (syntax-rules ()
    [(while-less e1 do e2)
     (letrec ([bound e1]
            [f (lambda () (let ([x e2])
                            (if (>= x bound)
                                #t
                                (f))
                            )
                 )])
       (f))]
    )
  )
                       