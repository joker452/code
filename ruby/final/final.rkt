#lang racket
;1
(define (partition f l)
  (if (null? l)
      (cons '() '())
      (let ([p (partition f (cdr l))]
            [x (car l)])
        (if (f x)
            (cons (cons x (car p)) (cdr p))
            (cons (car p) (cons x (cdr p)))
            )
        )
      )
  )

(define (number-or-not l)
  (partition number? l)
  )

; 1.(c) Yes (car (number-or-not xs)) is the list that holds all the numbers in xs, and so is
; (car (number-or-not (car (number-or-not xs))))

; 1.(d) Depends on f and/or xs.  If f is a pure function, then it's equivalent, otherwise, it's not.

; 2.
; (a) a1 = 9
; (b) a2 = 10
; (c) a3 = 7
; (d) a4 = 9
; (e) a5 = 11 -> 10
; (f) a6 = 12 -> 11
; (g) a7 = 7 -> 8
; (h) a8 = 9

(struct inftree (left-th root right-th))

(define (doubling-tree x)
  (letrec ([f (lambda(x) (inftree (lambda() (f (* x 2)))
                                  x
                                  (lambda() (f (+ (* x 2) 1)))
                                  ))])
    (lambda() (f x))
    )
  )

(define (sum-to-depth i t)
  (let ([x (t)])
    (if (= i 1)
        (inftree-root x)
        (+ (sum-to-depth (- i 1) (inftree-left-th x))
           (inftree-root x)
           (sum-to-depth (- i 1) (inftree-right-th x)))
        )
    )
  )

; 28
(define _x (sum-to-depth 3 (doubling-tree 1)))

(define (stream-of-lefts t)
  (letrec ([f (lambda(t) (let ([x (t)])
                           (cons (inftree-root x) (lambda() (f (inftree-left-th x))))
                           )
                )])
    (lambda() (f t))
    )
  )

;3. (e) (stream-of-lefts (doubling-tree 1)) is the stream of power of two, starting from 1

(define (envremove env s)
  (let ([f (lambda(x) (not (string=? s (car x))))])
    (filter f env)
    )
  )

(struct var (string) #:transparent)
(struct mlet (var e body) #:transparent)
(struct letinstead (varnew varold body) #:transparent)
(struct letalso (varnew varold body) #:transparent)

(define (eval-under-env e env)
  (cond [(letinstead? e) (let* ([varnew (letinstead-varnew e)]
                                [varold (letinstead-varold e)]
                                [body (letinstead-body e)]
                                [x (assoc varold env)])
                           (if x
                               (eval-under-env body (cons (cons varnew (cdr x))
                                                          (envremove env varold)))
                               (error "varold not in environment!")
                               )
                           )]
        [(letalso? e) (let* ([varnew (letalso-varnew e)]
                             [varold (letalso-varold e)]
                             [body (letalso-body e)]
                             [x (assoc varold env)])
                        (if x
                            (eval-under-env body (cons (cons varnew (cdr x)) env))
                            (error "varold not in environment!")
                            )
                        )]
        [#t (error "this sucks!")])
  )

(define (m varnew varold body)
  (mlet (var varnew) (var varold) body)
  )

;4. (e) mlet can only add local bindings, it can't delete bindings already existed

;5.
; (a) C
; (b) B
; (c) A
; (d) C
; (e) B

;6.
; (a) impossible
; (b) possible
; (c) impossible
; (d) possible


   