module Main where
import AST
import EvalType (evalType)
import           EvalValue                      ( evalValue )

p = Program [ADT "int_pair" [("P",[TInt,TInt])]] (ELet ("p",EApply (EApply (EVar "P") (EIntLit 10)) (EIntLit 20)) (ECase (EVar "p") [(PData "P" [PVar "x",PVar "x"],EAdd (EVar "x") (EVar "x"))]))

main :: IO ()
main = print $ evalType p
