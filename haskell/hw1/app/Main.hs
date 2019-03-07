module Main where

import Distance
main :: IO()
result = solution 3 (1, 2) (2,4)
main = do
    putStr (show result)

