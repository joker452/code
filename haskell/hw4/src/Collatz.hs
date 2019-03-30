module Collatz where

collatzs :: Integer -> [Integer]

collatzs x | x <= 0 = error "invalid input"
           | x == 1 = [1]
           | even x = x : collatzs (x `div` 2)
           | otherwise = x : collatzs (3 * x + 1)

