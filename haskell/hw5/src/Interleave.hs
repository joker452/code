module Interleave (
    interleave, interleaveLists)where

interleave :: [a] -> [a] -> [a]
interleave xs [] = xs
interleave [] ys = ys
interleave (x: xs) (y: ys) = x: y: interleave xs ys

-- remove the first element of the first n list in a list of list
updateList :: [[a]]  -> Integer -> [[a]]
updateList [] _ = []
updateList (xs: xss) n | n > 0 = tail xs: updateList xss (n - 1)
                   | otherwise = xs: xss


-- get the elemens in a list of list in anti-diagonal order, always from up to down, right to left
get :: [[a]] -> [[a]] -> Integer -> Integer -> [a]

get xss (xs: xss') n n'| n' > 0 = head xs: get xss xss' n (n' - 1)
                     | otherwise = let xss'' = updateList xss n in get xss'' xss'' (n + 1) (n + 1)

interleaveLists :: [[a]] -> [a]
interleaveLists xss = get xss xss 1 1
