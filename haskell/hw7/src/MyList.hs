module MyList (
    )where
import MyListType

instance Eq a => Eq (List a) where
    Nil == Nil = True
    (x:~ xs) == (y:~ ys) = x == y && xs == ys
    _xs == _ys = False


instance Ord a => Ord (List a) where
    compare Nil Nil = EQ
    compare (x:~ xs) Nil = GT
    compare Nil (y:~ ys) = LT
    compare (x:~ xs) (y:~ ys) | x == y = compare xs ys
                              | x <= y = LT
                              | otherwise = GT

showNonEmpty :: Show a => List a -> String
showNonEmpty (x:~ Nil) = show x ++ "]"
showNonEmpty (x:~ xs) = show x ++ "," ++ showNonEmpty xs

instance Show a => Show (List a) where
    show Nil = "[]"
    show xs = "[" ++ showNonEmpty xs

f :: Integer -> List Integer
f n = if n > 0 then n:~ f (n - 1) else Nil