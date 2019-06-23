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
  | VAdt String [Value]
  | VAdtFun String [Value] Int
  deriving (Show, Eq)

data Context = Context { getContext :: M.Map String Value } deriving (Show, Eq)

type ContextState a = StateT Context Maybe a

getCtorValue :: (String, [Type]) -> (String, Value)
getCtorValue (ctorName, []) = (ctorName, VAdt ctorName [])
getCtorValue (ctorName, pts) = (ctorName, VAdtFun ctorName [] (length pts))

getADTCtors :: [ADT] -> M.Map String Value
getADTCtors adts = M.fromList $ Prelude.concatMap (\adt -> case adt of (ADT adtName ctors) -> Prelude.map getCtorValue ctors) adts

matchPatterns :: [Pattern] -> [Value] -> Maybe (M.Map String Value)
matchPatterns [] [] = Just M.empty
matchPatterns _ [] = Nothing
matchPatterns [] _ = Nothing
matchPatterns [p] [v] = evalPV p v
                           
matchPatterns (p: ps) (v: vs) = do let rv = evalPV p v
                                   case rv of Just bindMap -> do let rv' = matchPatterns ps vs
                                                                 case rv' of Just bindsMap -> Just $ M.union bindsMap bindMap
                                                                             Nothing -> Nothing
                                              Nothing -> Nothing 

evalPV :: Pattern -> Value -> Maybe (M.Map String Value)
evalPV p v = case p of PBoolLit b -> case v of VBool b' -> if b == b' then Just M.empty else Nothing
                                               _ -> Nothing
                       PIntLit i -> case v of VInt i' -> if i == i' then Just M.empty else Nothing
                                              _ -> Nothing
                       PCharLit c -> case v of VChar c' -> if c == c' then Just M.empty else Nothing
                                               _ -> Nothing
                       PVar x -> Just $ M.fromList [(x, v)]
                       PData ctorName ps -> case v of VAdt ctorName' vs -> if ctorName == ctorName' then matchPatterns ps vs else Nothing
                                                      _ -> Nothing

evalCases :: Value -> [(Pattern, Expr)] -> ContextState Value
evalCases v [(p, e)] = do let rv = evalPV p v
                          case rv of Just bindings -> do oldContext <- get
                                                         let oldMap = getContext oldContext
                                                             newContext = Context $ M.union bindings oldMap
                                                         put newContext
                                                         v <- eval e
                                                         put oldContext
                                                         return v
                                     Nothing -> lift Nothing
evalCases v ((p, e): ps) = do let rv = evalPV p v
                              case rv of Just bindings -> do oldContext <- get
                                                             let oldMap = getContext oldContext
                                                                 newContext = Context $ M.union bindings oldMap
                                                             put newContext
                                                             v <- eval e 
                                                             put oldContext
                                                             return v
                                         Nothing -> evalCases v ps
 
eval :: Expr -> ContextState Value
eval (EBoolLit b) = return $ VBool b
eval (EIntLit i) = return $ VInt i
eval (ECharLit c) = return $ VChar c
eval (ENot e) = do (VBool b) <- eval e
                   return (VBool $ not b)
eval (EAnd e1 e2) = do (VBool b) <- eval e1
                       if b then eval e2 else return (VBool False)
eval (EOr e1 e2) = do (VBool b) <- eval e1
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
-- the binding is eval eagerly
-- add binding value to context first and restore afterwards
eval (ELet (n, e1) e2) = do oldContext <- get
                            v1 <- eval e1
                            let oldMap = getContext oldContext
                                newContext = Context $ M.insert n v1 oldMap
                            put newContext
                            v <- eval e2
                            put oldContext
                            return v
-- note that since function binding may be recursive
-- and when function is applied to expr, the context under which it's evaluated
-- is the one when it's defined, i.e., the function binding isn't included
-- so we need to add the biding manually every time before evaluate the recursive function
-- for this puporse, an additional function name variable is saved
eval (ELetRec f (x, tx) (e1, ty) e2)  = do oldContext <- get
                                           let oldMap = getContext oldContext
                                               newContext = Context $ M.insert f (VFun f oldContext x e1) oldMap
                                           put newContext
                                           v <- eval e2
                                           put oldContext
                                           return v            
eval (EVar n) = do oldContext <- get
                   let oldMap = getContext oldContext
                       v = M.lookup n oldMap
                   lift v

-- call by value
-- if it' not a recursive function, simply add the parameter value before eval body
-- otherwise the function binding itself should also be added
-- note union is left-biased, so the new bindng should come first to get correct result
-- when there is shadowing
eval (EApply e1 e2) = do f <- eval e1
                         v2 <- eval e2
                         oldContext <- get
                         (case f of  (VClosure ctx pn body) -> do let oldMap = getContext ctx
                                                                      newContext = Context $ M.insert pn v2 oldMap
                                                                  put newContext
                                                                  v <- eval body
                                                                  put oldContext
                                                                  return v
                                     (VFun name ctx pn body) -> do let oldMap = getContext ctx
                                                                       newContext = Context $ M.union (M.fromList [(pn, v2), (name, f)]) oldMap
                                                                   put newContext
                                                                   v <- eval body
                                                                   put oldContext
                                                                   return v
                                     (VAdtFun ctorName vs n) -> if n == 1 then return $ VAdt ctorName $ reverse (v2: vs) else return $ VAdtFun ctorName (v2: vs) (n - 1)
                                     _ -> lift Nothing)

eval (ECase e0 []) = lift Nothing
eval (ECase e0 ps) = do v0 <- eval e0
                        evalCases v0 ps
                        

evalProgram :: Program -> Maybe Value
evalProgram (Program adts body) = evalStateT (eval body) $
  Context $ getADTCtors adts 


evalValue :: Program -> Result
evalValue p = case evalProgram p of
  Just (VBool b) -> RBool b
  Just (VInt i) -> RInt i
  Just (VChar c) -> RChar c
  _ -> RInvalid
