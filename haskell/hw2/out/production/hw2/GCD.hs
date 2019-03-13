module GCD
    (myGCD
    )where

myGCD :: Integer -> Integer -> Integer
myGCD _ 0 = 0
myGCD 0 _ = 0
myGCD a b
    | x == y =  x
    | x `mod` 2 == 0 && y `mod` 2 == 0 = 2 * myGCD (x `div` 2) (y `div` 2)
    | x `mod` 2 == 0 && y `mod` 2 /= 0 = myGCD (x `div` 2) y
    | x `mod` 2 /= 0 && y `mod` 2 == 0 = myGCD x (y `div` 2)
    | otherwise = if x > y  then myGCD ((x - y) `div` 2) y else myGCD ((y - x) `div` 2) x
    where x = abs a
          y = abs b


