module MyListType where

data List a = a :~ (List a) | Nil
infixr 5 :~