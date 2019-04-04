module Interleave where

interleave :: [a] -> [a] -> [a]
interleave xs [] = xs
interleave [] ys = ys
interleave (x: xs) (y: ys) = x: y: interleave xs ys

helper :: [[a]]  -> Integer -> [[a]]
helper [] _ = []
helper (xs: xss) n | n > 0 = tail xs: helper xss (n - 1)
                   | otherwise = (xs: xss)



h :: [[a]] -> [[a]] -> Integer -> Integer -> [a]

h xss (xs: xss') n n'| n' > 0 = head xs: h xss xss' n (n' - 1)
                     | otherwise = let xss'' = helper xss n in h xss'' xss'' (n + 1) (n + 1)

interleaveLists :: [[a]] -> [a]
interleaveLists xss = h xss xss 1 1
