{-# LANGUAGE InstanceSigs #-}
module Compose where

newtype Compose f g a = Compose { getCompose :: f (g a) } deriving (Eq, Show)

instance (Functor f, Functor g) => Functor (Compose f g) where
    fmap :: (a -> b) -> Compose f g a -> Compose f g b
    fmap f (Compose fga) = Compose $ (fmap . fmap) f fga

instance (Applicative f, Applicative g) => Applicative (Compose f g) where
    pure :: a -> Compose f g a
    pure a = Compose $ (pure . pure) a

    (<*>) :: Compose f g (a -> b) -> Compose f g a -> Compose f g b
    -- type of fgab should be f g xxx -> xxx
    -- type of fga should be f g xxx
    -- 1st <*>: g (a -> b) -> g a -> g b
    -- <$>: (g a -> g b) -> f (g a) -> f (g b)
    -- 2nd <*>: f (g a -> g b) -> f (g a) -> f g b
    (Compose fgab) <*> (Compose fga) = Compose $ ((<*>) <$> fgab) <*> fga


