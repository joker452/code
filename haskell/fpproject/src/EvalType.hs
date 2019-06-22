-- | 这是其中一种实现方式的代码框架。你可以参考它，或用你自己的方式实现，只要按需求完成 evalType :: Program -> Maybe Type 就行。
module EvalType where 

import AST
import Data.Map.Strict as M
import Control.Monad.State



data Context = Context { getContext :: M.Map String Type }
  deriving (Show, Eq)

type ContextState a = StateT Context Maybe a



eval :: Expr -> ContextState Type
eval (EBoolLit _) = return TBool
eval (EIntLit _) =  return TInt
eval (ECharLit _) = return TChar
eval (ENot e) = do t <- eval e
                   if t == TBool then return TBool else lift Nothing
eval (EAnd e1 e2) = do t1 <- eval e1
                       t2 <- eval e2
                       if t1 == TBool && t2 == TBool then return TBool else lift Nothing
eval (EOr e1 e2) = do t1 <- eval e1
                      t2 <- eval e2
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
-- before check the type of function body, 
-- we need to add the parameter type to the context and restore afterwards                     
eval (ELambda (pn, pt) e) = do oldContext <- get
                               let oldMap = getContext oldContext
                                   newContext = Context $ M.insert pn pt oldMap
                               put newContext
                               rt <- eval e
                               put oldContext
                               return $ TArrow pt rt 
-- same as above, need to add bind type first and restore afterwards                               
eval (ELet (n, e1) e2) =  do t1 <- eval e1
                             oldContext <- get
                             let oldMap = getContext oldContext
                                 newContext = Context $ M.insert n t1 oldMap
                             put newContext
                             t2 <- eval e2
                             put oldContext
                             return t2
-- similar to above, note that since function binding may be recursive
-- we need to add the declared function type before actually check the type
-- the actual type should match the declared one                              
eval (ELetRec f (x, tx) (e1, ty) e2) = do oldContext <- get
                                          let oldMap = getContext oldContext
                                              newContext = Context $ M.insert f (TArrow tx ty) oldMap
                                          put newContext
                                          tLambda <- eval $ ELambda (x, tx) e1
                                          if tLambda == TArrow tx ty then do t <- eval e2
                                                                             put oldContext
                                                                             return t
                                                                          else do put oldContext
                                                                                  lift Nothing
                                                                         

eval (EVar n) = do context <- get
                   let oldMap = getContext context 
                       t = M.lookup n oldMap
                   lift t
-- paramter type should match argument type
eval (EApply e1 e2) = do tArrow <- eval e1
                         pt <- eval e2
                         (case tArrow of TArrow t0 t1 -> if t0 == pt then return t1 else lift Nothing
                                         _ -> lift Nothing)                  
-- ... more
eval _ = undefined

evalType :: Program -> Maybe Type
evalType (Program adts body) = evalStateT (eval body) $
  Context $ (M.empty :: M.Map String Type)  -- 可以用某种方式定义上下文，用于记录变量绑定状态
