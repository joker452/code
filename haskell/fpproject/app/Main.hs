module Main where
import AST
import EvalType (evalType)
import           EvalValue                      ( evalValue )
callFun :: Expr -> [Expr] -> Expr
callFun f [e     ] = EApply f e
callFun f (e : es) = callFun (EApply f e) es
p = Program [] $ ELet ("zero", ELambda ("f", TArrow TInt TInt) (ELambda ("x", TInt) (EVar "x"))) 
      $ ELet ("f", ELambda ("x", TInt) (EAdd (EVar "x") (EIntLit 1))) 
      $   ELet ("succ", ELambda ("n", TArrow (TArrow TInt TInt) (TArrow TInt TInt))
      (ELambda ("f", TArrow TInt TInt)
        (ELambda ("x", TInt)
          (EApply (EVar "f")
            (callFun (EVar "n") [EVar "f", EVar "x"])))))
      $ ELet ("one", EApply (EVar "succ") (EVar "zero"))
      $ callFun (EVar "zero") [EVar "f", EIntLit 1]

main :: IO ()
main = do print $ evalValue p
          print $ evalType p
