;; CSE341, Programming Languages, Homework 5

#lang racket
(provide (all-defined-out)) ;; so we can put tests in a second file

;; definition of structures for MUPL programs - Do NOT change
(struct var  (string) #:transparent)  ;; a variable, e.g., (var "foo")
(struct int  (num)    #:transparent)  ;; a constant number, e.g., (int 17)
(struct add  (e1 e2)  #:transparent)  ;; add two expressions
(struct ifgreater (e1 e2 e3 e4)    #:transparent) ;; if e1 > e2 then 1 else 0
(struct fun  (nameopt formal body) #:transparent) ;; a recursive(?) 1-argument function
(struct call (funexp actual)       #:transparent) ;; function call
(struct mlet (var e body) #:transparent) ;; a local binding (let var = e in body) 
(struct apair   (e1 e2) #:transparent) ;; make a new pair
(struct fst   (e)     #:transparent) ;; get first part of a pair
(struct snd (e)     #:transparent) ;; get second part of a pair
(struct aunit   ()      #:transparent) ;; unit value -- good for ending a list
(struct isaunit (e)     #:transparent) ;; if e1 is unit then 1 else 0

;; a closure is not in "source" programs; it is what functions evaluate to
(struct closure (env fun) #:transparent) 

;; Problem 1
(define (racketlist->mupllist rlist)
  (cond [(null? rlist) (aunit)]
        [(list? rlist) (apair (car rlist) (racketlist->mupllist (cdr rlist)))]
        [#t (error "racketlist->mupllist applied to non-list")]
        )
  )

(define (mupllist->racketlist mlist)
  (cond [(aunit? mlist) null]
        [(apair? mlist) (cons (apair-e1 mlist) (mupllist->racketlist (apair-e2 mlist)))]
        [#t (error "mupllist->racketlist applied to non-apair")]
        )
  )


;; CHANGE (put your solutions here)

;; Problem 2

;; lookup a variable in an environment
;; Do NOT change this function
(define (envlookup env str)
  (cond [(null? env) (error "unbound variable during evaluation" str)]
        [(equal? (car (car env)) str) (cdr (car env))]
        [#t (envlookup (cdr env) str)]))

;; Do NOT change the two cases given to you.  
;; DO add more cases for other kinds of MUPL expressions.
;; We will test eval-under-env by calling it directly even though
;; "in real life" it would be a helper function of eval-exp.
(define (eval-under-env e env)
  (cond [(var? e) 
         (envlookup env (var-string e))]
        [(add? e) 
         (let ([v1 (eval-under-env (add-e1 e) env)]
               [v2 (eval-under-env (add-e2 e) env)])
           (if (and (int? v1)
                    (int? v2))
               (int (+ (int-num v1) 
                       (int-num v2)))
               (error "MUPL addition applied to non-number")))]
        ;; CHANGE add more cases here
        ;; mupl values
        [(int? e) e]
        [(closure? e) e]
        [(aunit? e) e]
        [(fun? e) (closure env e)]
        [(ifgreater? e)
         (let ([v1 (eval-under-env (ifgreater-e1 e) env)]
               [v2 (eval-under-env (ifgreater-e2 e) env)])
           (if (and (int? v1) (int? v2))
               (if (> (int-num v1) (int-num v2))
                   (eval-under-env (ifgreater-e3 e) env)
                   (eval-under-env (ifgreater-e4 e) env)
                   )
               (error "MUPL ifgreater applied to non-number")
               )
           )]
        [(mlet? e)
         (let ([v1 (eval-under-env (mlet-e e) env)])
           (eval-under-env (mlet-body e) (cons (cons (mlet-var e) v1) env))
           )]
        [(call? e)
         (let ([f (eval-under-env (call-funexp e) env)]
               [actual (eval-under-env (call-actual e) env)])
           (if (closure? f)
               (let ([name (fun-nameopt (closure-fun f))]
                     [formal (fun-formal (closure-fun f))]
                     [body (fun-body (closure-fun f))])
                 ;; function with a name or anonymous function
                 (if name
                     (eval-under-env body (cons (cons name f) (cons (cons formal actual) (closure-env f))))
                     (eval-under-env body (cons (cons formal actual) (closure-env f)))
                     )
                 )
               (error "MUPL call applied to non-closure")
               )
           )]
        [(apair? e)
         (let ([v1 (eval-under-env (apair-e1 e) env)]
               [v2 (eval-under-env (apair-e2 e) env)])
           (apair v1 v2)
           )]
        [(fst? e)
         (let ([v (eval-under-env (fst-e e) env)])
           (if (apair? v)
               (apair-e1 v)
               (error "MUPL fst applied to non-apair")
               )
           )]
        [(snd? e)
         (let ([v (eval-under-env (snd-e e) env)])
           (if (apair? v)
               (apair-e2 v)
               (error "MUPL fst applied to non-apair")
               )
           )]
        [(isaunit? e)
         (let ([v (eval-under-env (isaunit-e e) env)])
           (if (aunit? v)
               (int 1)
               (int 0)
               )
           )]
        [#t (error (format "bad MUPL expression: ~v" e))]))

;; Do NOT change
(define (eval-exp e)
  (eval-under-env e null))
        
;; Problem 3

(define (ifaunit e1 e2 e3)
  (ifgreater (isaunit e1) (int 0) e2 e3)
  )

(define (mlet* bs e2)
  (if (null? bs)
      e2
      (mlet (car (car bs)) (cdr (car bs)) (mlet* (cdr bs) e2))
      )
  )

(define (ifeq e1 e2 e3 e4)
  (mlet "_x" e1
        (mlet "_y" e2
              (ifgreater (var "_x") (var "_y") e4
                         (ifgreater (var "_y") (var "_x") e4 e3)
                         )
              )
        )
  )
;; Problem 4

(define mupl-map
  (fun "f-outer" "map-function"
       (fun "f-inner" "l"
            (ifaunit (var "l") (aunit)
                  (apair (call (var "map-function") (fst (var "l"))) (call (var "f-inner") (snd (var "l")))))
            )
       )
  )

(define mupl-mapAddN
  (mlet "map" mupl-map
        (fun "map-add" "i"
             (call (var "map") (fun "add" "x" (add (var "x") (var "i")))))
        )
  )

;; Challenge Problem

(struct fun-challenge (nameopt formal body freevars) #:transparent) ;; a recursive(?) 1-argument function

;; We will test this function directly, so it must do
;; as described in the assignment
(define (compute-free-vars e) "CHANGE")

;; Do NOT share code with eval-under-env because that will make grading
;; more difficult, so copy most of your interpreter here and make minor changes
(define (eval-under-env-c e env) "CHANGE")

;; Do NOT change this
(define (eval-exp-c e)
  (eval-under-env-c (compute-free-vars e) null))