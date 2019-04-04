module Bachelor (
    bachelor)where

split' :: [a] -> ([a], [a])
split' [] = ([], [])
split' [a] = ([a], [])
split' (x:y:xs) = (x: xs', y: ys')
               where (xs', ys') = split' xs

merge :: Ord a => [a] -> [a] -> [a]
merge [] ys = ys
merge xs [] = xs
merge (x: xs) (y: ys) | x < y = x: merge xs (y: ys)
                      | otherwise = y: merge (x: xs) ys

mergeSort :: Ord a => [a] -> [a]
mergeSort [] = []
mergeSort [x] = [x]
mergeSort xs = merge (mergeSort l) (mergeSort r)
            where (l, r) = split' xs

helper :: [Integer] -> Integer
helper [x] = x
helper (x: y: xs) | x == y = helper xs
                  | otherwise = x

bachelor :: [Integer] -> Integer
bachelor [x] = x
bachelor xs = helper xs'
           where xs' = mergeSort xs
