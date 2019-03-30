module Brute (getPINs
    )where
import Data.Char


genList :: Char -> String
genList c | c' == 0 = ['0', '8']
          | c' == 2 = ['1', '2', '3', '5']
          | c' == 5 = ['2', '4', '5', '6', '8']
          | c' == 8 = ['0', '5', '7', '8', '9']
          | c' `elem` [1, 3] = map intToDigit [2, c', c' + 3]
          | c' `elem` [7, 9] = map intToDigit [8, c' - 3, c']
          | c' `elem` [4, 6] = map intToDigit [5, c', c' - 3, c' + 3]
          | otherwise = error "invalid input"
          where c' = digitToInt c

getPINs :: String -> [String]
getPINs  = mapM genList