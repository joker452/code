{-# LANGUAGE InstanceSigs #-}
module Reader where

import           Data.Char

cap :: String -> String
cap = map toUpper

rev :: String -> String
rev  = reverse

composed :: String -> String
composed = cap . rev

fmapped :: String -> String
fmapped = cap <$> rev

tupled :: String -> (String, String)
tupled = (,) <$> cap <*> rev

tupled' :: String -> (String, String)
tupled' = do
    x <- cap
    y <- rev
    return (x, y)

tupled'' :: String -> (String, String)
tupled'' = cap >>= (\x -> rev >>= (\y -> return (x, y)))

newtype Reader r a = Reader { runReader :: r -> a}
ask :: Reader a a
ask = Reader id

myLiftA2 :: Applicative f => (a -> b -> c) -> f a -> f b -> f c
myLiftA2 f x y = f <$> x <*> y

asks :: (r -> a) -> Reader r a
asks = Reader

instance Functor (Reader a) where
    fmap f (Reader x) = Reader $ f . x


instance Applicative (Reader r) where
    pure :: a -> Reader r a
    pure a = Reader $ const a
    (<*>) :: Reader r (a -> b) -> Reader r a -> Reader r b
    (Reader rab) <*> (Reader ra) = Reader $ \r -> rab r (ra r)

instance Monad (Reader r) where
    return = pure
    (>>=) :: Reader r a -> (a -> Reader r b) -> Reader r b
    (Reader ra) >>= aRb = Reader $ \r -> runReader (aRb $ ra r) r

