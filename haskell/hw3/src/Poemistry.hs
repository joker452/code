module Poemistry(poemistry, prettyPrint) where

helper :: [Char] -> Integer -> Integer -> Integer -> [Char]
helper xs len k acc | acc >= 20 = []
                 | otherwise = xs !! fromInteger (k `div` len ^ (19 - acc)) : helper xs len (k `mod` len ^ (19 - acc)) (acc + 1)

poemistry :: [Char] -> Integer -> [Char]

poemistry xs k = helper xs (toInteger (length xs)) k 0

prettyPrint :: [Char] -> [Char]
prettyPrint xs | length xs /= 20 = error "only char list of length 20 is permitted"
               | otherwise = let (x5, x6) = splitAt 5 xs in
                                  let (x10, x11) = splitAt 5 x6 in
                                       let (x15, x16) = splitAt 5 x11 in x5 ++ "\n" ++ x10 ++ "\n" ++ x15 ++ "\n" ++ x16 ++ "\n"
