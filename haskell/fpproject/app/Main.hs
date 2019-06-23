module Main where
import AST
import EvalType (evalType)
import EvalValue (evalProgram)

list = (EApply (EApply (EVar "Cons") (EIntLit 1)) (EApply (EApply (EVar "Cons") (EIntLit 2)) ((EApply (EApply (EVar "Cons") (EIntLit 3)) (EVar "Nil")))))
list12 = (EApply (EApply (EVar "Cons") (EIntLit 1)) ((EApply (EApply (EVar "Cons") (EIntLit 2)) (EVar "Nil"))))
list13 = (EApply (EApply (EVar "Cons") (EIntLit 1)) ((EApply (EApply (EVar "Cons") (EIntLit 3)) (EVar "Nil"))))
list1 = ((EApply (EApply (EVar "Cons") (EIntLit 1)) (EVar "Nil")))



safeDiv = ELambda ("x",TInt) (ELambda ("y",TInt) (EIf (EEq (EVar "y") (EIntLit 0)) (EVar "Nothing") (EApply (EVar "JustInt") (EDiv (EVar "x") (EVar "y")))))

mapbody = ELetRec "map" ("xs",TData "Array") ((ECase (EVar "xs") [(PData "Nil" [],(EVar "Nil")),((PData "Cons" [PVar "x",PVar "xs1"]),EApply (EApply (EVar "Cons") (EApply (EVar "f") (EVar "x"))) (EApply (EVar "map") (EVar "xs1")))]),TData "Array") (EApply (EVar "map") (EVar "xss"))
map = ELambda  ("f",(TArrow TInt TInt )) (ELambda ("xss",TData "Array") mapbody) 
func = ELambda ("x",TInt) (EAdd (EVar "x") (EIntLit 1))
sqrt = ELambda ("x",TInt) (EMul (EVar "x") (EVar "x"))


filterExpr = (ECase (EVar "xs") 
                [
                    (
                        PData "Nil" [],
                        (EVar "Nil")
                    ),
                    (
                        (PData "Cons" [PVar "x",PVar "xs1"]),
                        EIf 
                            (EApply (EVar "f") (EVar "x")) 
                            (EApply (EApply (EVar "Cons") (EVar "x")) (EApply (EVar "filter") (EVar "xs1")))
                            (EApply (EVar "filter") (EVar "xs1"))
                    )
                ])
filterbody = ELetRec "filter" ("xs",TData "Array") (filterExpr,TData "Array") (EApply (EVar "filter") (EVar "xss"))
filter = ELambda  ("f",(TArrow TInt TBool )) (ELambda ("xss",TData "Array") filterbody) 
filterFunc = ELambda ("x",TInt) (EGt (EVar "x") (EIntLit 1))
even = ELambda ("x",TInt) (EEq (EMod (EVar "x") (EIntLit 2)) (EIntLit 0))






equalExpr = ELambda ("ys",TData "Array") (ECase (EVar "xs") 
                [
                    (
                        (PData "Nil" []),
                        (ECase (EVar "ys") 
                        [
                            (
                                (PData "Nil" []),
                                (EBoolLit True)
                            ),
                            (
                                (PData "Cons" [PVar "y",PVar "ys1"]),
                                (EBoolLit False)
                            )
                        ])
                    ),
                    (
                        (PData "Cons" [PVar "x",PVar "xs1"]),
                        (ECase (EVar "ys") 
                        [
                            (
                                (PData "Nil" []),
                                (EBoolLit False)
                            ),
                            (
                                (PData "Cons" [PVar "y",PVar "ys1"]),
                                EIf 
                                    (EEq (EVar "x") (EVar "y"))
                                    (EApply (EApply (EVar "equal") (EVar "xs1")) (EVar "ys1"))
                                    (EBoolLit False)
                            )
                        ])
                    )
                ])
equalbody = ELetRec "equal" ("xs",TData "Array") (equalExpr,(TArrow (TData "Array") TBool)) ((EApply (EApply (EVar "equal") (EVar "xss")) (EVar "yss")))
equal = ELambda ("xss",TData "Array") (ELambda ("yss",TData "Array") equalbody)

array = ADT "Array" [("Nil",[]),("Cons",[TInt ,(TData "Array")])]
maybeInt = ADT "MaybeInt" [("JustInt",[TInt]),("Nothing",[])]


p0 = Program [array,maybeInt] (EApply (EApply safeDiv (EIntLit 7)) (EIntLit 0))
p1 = Program [array,maybeInt] (EApply (EApply safeDiv (EIntLit 7)) (EIntLit 2))
p2 = Program [array,maybeInt] (EApply (EApply Main.map Main.sqrt) list)
p3 = Program [array,maybeInt] (EApply (EApply Main.filter Main.even) list)
p4 = Program [array,maybeInt] (EApply (EApply equal (EVar "Nil")) (EVar "Nil"))
p5 = Program [array,maybeInt] (EApply (EApply equal list12) list12)
p6 = Program [array,maybeInt] (EApply (EApply equal list12) list13)
p7 = Program [array,maybeInt] (EApply (EApply equal list12) list1)
main :: IO ()
main = do print $ evalType p7
          print $ evalProgram p7
