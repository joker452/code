module MyList where

import Prelude hiding (concat)

data List a = a :~ (List a) | Nil
    deriving (Show)
infixr 5 :~

concat :: List (List a) -> List a
concat Nil = Nil
concat ((:~) x xs) = case xs of
    Nil -> concat x
    _ -> concat x ++ concat xs

