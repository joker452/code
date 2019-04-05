module Bachelor (
    bachelor)where
import Data.List(sort)

compareHelper :: [Integer] -> Integer
compareHelper [x] = x
compareHelper (x: y: xs) | x == y = compareHelper xs
                  | otherwise = x

bachelor :: [Integer] -> Integer
bachelor [x] = x
bachelor xs = compareHelper xs'
           where xs' = sort xs
