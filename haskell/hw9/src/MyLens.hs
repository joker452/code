{-# LANGUAGE RankNTypes #-}
module MyLens
  ( Lens
  , over
  , view
  , (%~)
  , (^.)
  ) where

-- Identity
newtype Identity a = Identity { runIdentity :: a }

instance Functor Identity where
    fmap f (Identity x) = Identity (f x)

-- Const
newtype Const a b = Const { getConst :: a }

instance Functor (Const a) where
    fmap _ (Const a) = Const a

-- Lens
type Lens b a = forall f . Functor f => (a -> f a) -> b -> f b

over :: Lens b a -> (a -> a) -> (b -> b)
over l f = runIdentity . l (Identity . f)

view :: Lens b a -> b -> a
view l = getConst . l Const

(%~) :: Lens b a -> (a -> a) -> (b -> b)
(%~) = over
infixr 4 %~

(^.) :: b -> Lens b a -> a
b ^. lens = view lens b
infixl 8 ^.
