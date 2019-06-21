-- | 这是其中一种实现方式的代码框架。你可以参考它，或用你自己的方式实现，只要按需求完成 evalValue :: Program -> Result 就行。
module EvalValue where

import AST
import Data.Map.Strict as M
import Debug.Trace
import Control.Monad.State

data Value
  = VBool Bool
  | VInt Int
  | VChar Char
  | VClosure Context String Expr
  | VFun String Context String Expr
  -- ... more
  deriving (Show, Eq)

data Context = Context { getContext :: M.Map String Value } deriving (Show, Eq)

type ContextState a = StateT Context Maybe a

gg :: Value -> String
gg (VClosure _ s _) = s
getBool :: Expr -> ContextState Bool
getBool e = do
  ev <- eval e
  case ev of
    VBool b -> return b
    _ -> lift Nothing

eval :: Expr -> ContextState Value
eval (EBoolLit b) = return $ VBool b
eval (EIntLit i) = return $ VInt i
eval (ECharLit c) = return $ VChar c
eval (ENot e) = getBool e >>= \b -> return (VBool $ not b)
eval (EAnd e1 e2) = do b <- getBool e1
                       if b then eval e2 else return (VBool False)
eval (EOr e1 e2) = do b <- getBool e1
                      if not b then eval e2 else return (VBool True)
eval (EAdd e1 e2) = do (VInt i1) <- eval e1
                       (VInt i2) <- eval e2
                       return $ VInt $ i1 + i2
eval (ESub e1 e2) = do (VInt i1) <- eval e1
                       (VInt i2) <- eval e2
                       return $ VInt $ i1 - i2
eval (EMul e1 e2) = do (VInt i1) <- eval e1
                       (VInt i2) <- eval e2
                       return $ VInt $ i1 * i2
eval (EDiv e1 e2) = do (VInt i1) <- eval e1
                       (VInt i2) <- eval e2
                       if i2 == 0 then lift Nothing else return $ VInt $ div i1 i2
eval (EMod e1 e2) = do (VInt i1) <- eval e1
                       (VInt i2) <- eval e2
                       if i2 == 0 then lift Nothing else return $ VInt $ mod i1 i2
eval (EEq e1 e2) = do v1 <- eval e1
                      v2 <- eval e2
                      (case v1 of VInt i1 -> let (VInt i2) = v2 in return $ VBool $ i1 == i2
                                  VBool b1 -> let (VBool b2) = v2 in return $ VBool $ b1 == b2
                                  VChar c1 -> let (VChar c2) = v2 in return $ VBool $ c1 == c2
                                  _ -> lift Nothing)
eval (ENeq e1 e2) = do v1 <- eval e1
                       v2 <- eval e2
                       (case v1 of VInt i1 -> let (VInt i2) = v2 in return $ VBool $ i1 /= i2
                                   VBool b1 -> let (VBool b2) = v2 in return $ VBool $ b1 /= b2
                                   VChar c1 -> let (VChar c2) = v2 in return $ VBool $ c1 /= c2
                                   _ -> lift Nothing)
eval (ELt e1 e2) = do v1 <- eval e1
                      v2 <- eval e2
                      (case v1 of VInt i1 -> let (VInt i2) = v2 in return $ VBool $ i1 < i2
                                  VChar c1 -> let (VChar c2) = v2 in return $ VBool $ c1 < c2
                                  _ -> lift Nothing)
eval (EGt e1 e2) = do v1 <- eval e1
                      v2 <- eval e2
                      (case v1 of VInt i1 -> let (VInt i2) = v2 in return $ VBool $ i1 > i2
                                  VChar c1 -> let (VChar c2) = v2 in return $ VBool $ c1 > c2
                                  _ -> lift Nothing)
eval (ELe e1 e2) = do v1 <- eval e1
                      v2 <- eval e2
                      (case v1 of VInt i1 -> let (VInt i2) = v2 in return $ VBool $ i1 <= i2
                                  VChar c1 -> let (VChar c2) = v2 in return $ VBool $ c1 <= c2
                                  _ -> lift Nothing)
eval (EGe e1 e2) = do v1 <- eval e1
                      v2 <- eval e2
                      (case v1 of VInt i1 -> let (VInt i2) = v2 in return $ VBool $ i1 >= i2
                                  VChar c1 -> let (VChar c2) = v2 in return $ VBool $ c1 >= c2
                                  _ -> lift Nothing)
eval (EIf e1 e2 e3) = do (VBool b) <- eval e1
                         if b then eval e2 else eval e3

eval (ELambda (pn, _) e) = do context <- get
                              return $ VClosure context pn e
eval (ELet (n, e1) e2) = do context <- get
                            v1 <- eval e1
                            let oldMap = getContext context
                                newContext = Context $ M.insert n v1 oldMap
                            -- traceM ("in ELet " ++ n ++ show (toList oldMap))
                            put newContext
                            v <- eval e2
                            put context
                            return v
eval (ELetRec f (x, tx) (e1, ty) e2)  = do context <- get
                                           let oldMap = getContext context
                                               newContext = Context $ M.insert f (VFun f context x e1) oldMap
                                           put newContext
                                           v <- eval e2
                                           put context
                                           return v            
eval (EVar n) = do oldContext <- get
                   let oldMap = getContext oldContext
                       v = M.lookup n oldMap
                   lift v


eval (EApply e1 e2) = do f <- eval e1
                        --  traceM("f " ++ show f)
                         v2 <- eval e2
                        --  traceM("v2 " ++ show v2)
                         newContext <- get
                         (case f of  (VClosure oldContext pn e1') -> do let oldMap = getContext oldContext
                                                                            oldContext' = Context $ M.insert pn v2 oldMap
                                                                        put oldContext'
                                                                        -- traceM ("body " ++ show e1')
                                                                        v <- eval e1'
                                                                        put newContext
                                                                        return v
                                     (VFun f' oldContext pn e1') -> do let oldMap = getContext oldContext
                                                                           oldContext' = Context $ M.union (M.fromList [(pn, v2), (f', f)]) oldMap
                                                                       put oldContext'
                                                                       v <- eval e1'
                                                                       put newContext
                                                                       return v
                                     _ -> lift Nothing)




-- ... more
eval _ = undefined

evalProgram :: Program -> Maybe Value
evalProgram (Program adts body) = evalStateT (eval body) $
  Context (M.empty :: M.Map String Value) 


evalValue :: Program -> Result
evalValue p = case evalProgram p of
  Just (VBool b) -> RBool b
  Just (VInt i) -> RInt i
  Just (VChar c) -> RChar c
  _ -> RInvalid

                  --  (case v of Just v' -> case v' of (VClosure context n e) -> do put context
                  --                                                                traceM("In EVar context is " ++ show (Prelude.map fst (toList $ getContext context)))
                  --                                                                traceM("e " ++ show e)
                  --                                                                vFinal <- eval $ EApply e
                  --                                                                traceM("vFinal " ++ show vFinal)
                  --                                                                put oldContext
                  --                                                                return vFinal
                                                    
                  --                                   _ -> return v'
                  --             Nothing -> lift Nothing)