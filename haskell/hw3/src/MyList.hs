module MyList where

import Prelude hiding (concat)

data List a = a :~ (List a) | Nil
    deriving (Show)
infixr 5 :~

add :: List a -> List a -> List a
add Nil ys = ys
add xs Nil = xs
add (x :~ xs) ys = x :~ add xs ys

concat :: List (List a) -> List a
concat Nil = Nil
concat (x :~ xs) = add x (concat xs)

