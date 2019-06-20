-- | 这是其中一种实现方式的代码框架。你可以参考它，或用你自己的方式实现，只要按需求完成 evalType :: Program -> Maybe Type 就行。
module EvalType where 

import AST
import Data.Map.Strict as M
import Control.Monad.State



data Context = Context { getContext :: M.Map String Type }
  deriving (Show, Eq)

type ContextState a = StateT Context Maybe a
isBool :: Expr -> ContextState Type
isBool e = do
  et <- eval e
  case et of
    TBool -> return TBool
    _ -> lift Nothing


eval :: Expr -> ContextState Type
eval (EBoolLit _) = return TBool
eval (EIntLit _) =  return TInt
eval (ECharLit _) = return TChar
eval (ENot e) = isBool e >> return TBool
eval (EAnd e1 e2) = do t1 <- isBool e1
                       t2 <- isBool e2
                       if t1 == TBool && t2 == TBool then return TBool else lift Nothing
eval (EOr e1 e2) = do t1 <- isBool e1
                      t2 <- isBool e2
                      if t1 == TBool && t2 == TBool then return TBool else lift Nothing
eval (EAdd e1 e2) = do t1 <- eval e1
                       t2 <- eval e2
                       if t1 == TInt && t2 == TInt then return TInt else lift Nothing
eval (ESub e1 e2) = do t1 <- eval e1
                       t2 <- eval e2
                       if t1 == TInt && t2 == TInt then return TInt else lift Nothing   

eval (EMul e1 e2) = do t1 <- eval e1
                       t2 <- eval e2
                       if t1 == TInt && t2 == TInt then return TInt else lift Nothing  
eval (EDiv e1 e2) = do t1 <- eval e1
                       t2 <- eval e2
                       if t1 == TInt && t2 == TInt then return TInt else lift Nothing
eval (EMod e1 e2) = do t1 <- eval e1
                       t2 <- eval e2
                       if t1 == TInt && t2 == TInt then return TInt else lift Nothing
eval (EEq e1 e2) = do t1 <- eval e1
                      t2 <- eval e2
                      if (t1 == TInt || t1 == TBool || t1 == TChar) && t1 == t2 then return TBool else lift Nothing
eval (ENeq e1 e2) = do t1 <- eval e1
                       t2 <- eval e2
                       if (t1 == TInt || t1 == TBool || t1 == TChar) && t1 == t2 then return TBool else lift Nothing
eval (ELt e1 e2) = do t1 <- eval e1
                      t2 <- eval e2
                      if (t1 == TInt || t1 == TChar) && t1 == t2 then return TBool else lift Nothing
                      
eval (EGt e1 e2) = do t1 <- eval e1
                      t2 <- eval e2
                      if (t1 == TInt || t1 == TChar) && t1 == t2 then return TBool else lift Nothing
eval (ELe e1 e2) = do t1 <- eval e1
                      t2 <- eval e2
                      if (t1 == TInt || t1 == TChar) && t1 == t2 then return TBool else lift Nothing
eval (EGe e1 e2) = do t1 <- eval e1
                      t2 <- eval e2
                      if (t1 == TInt || t1 == TChar) && t1 == t2 then return TBool else lift Nothing

eval (EIf e1 e2 e3) = do t1 <- eval e1
                         t2 <- eval e2
                         t3 <- eval e3
                         if t1 == TBool && t2 == t3 then return t2 else lift Nothing
eval (ELambda (pn, pt) e) = do context <- get
                               let oldMap = getContext context
                                   oldType = M.lookup pn oldMap
                                   newContext = Context $ M.insert pn pt oldMap
                               put newContext
                               rt <- eval e
                               put $ Context (case oldType of
                                                            Just t -> M.insert pn t oldMap
                                                            Nothing -> M.delete pn oldMap) 
                               return $ TArrow pt rt 
                               
eval (ELet (n, e1) e2) =  do t1 <- eval e1
                             context <- get
                             let oldMap = getContext context
                                 newContext = Context $ M.insert n t1 oldMap
                             put newContext
                             t2 <- eval e2
                             put $ Context (M.delete n oldMap)
                             return t2

eval (EVar n) = do context <- get
                   let oldMap = getContext context 
                       t = M.lookup n oldMap
                   (case t of Just t -> return t
                              Nothing -> lift Nothing)
                                       
 
                               
                               
-- ... more
eval _ = undefined

evalType :: Program -> Maybe Type
evalType (Program adts body) = evalStateT (eval body) $
  Context $ (M.empty :: M.Map String Type)  -- 可以用某种方式定义上下文，用于记录变量绑定状态
