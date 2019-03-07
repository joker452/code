module LCM
    (solution
    )where
solution :: Integer -> Integer -> Integer

solution x y = x * y `div` gcd x y

