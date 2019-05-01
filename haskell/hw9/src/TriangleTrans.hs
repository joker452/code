module TriangleTrans where

import MyLens
import TriangleType

coordXLens :: Lens Point Integer
coordXLens f (Point x y) = fmap (\x' -> Point x' y) (f x)
coordYLens :: Lens Point Integer
coordYLens f (Point x y) = fmap (\y' -> Point x y') (f y)
triALens :: Lens Triangle Point
triALens f (Triangle a b c) = fmap (\a' -> Triangle a' b c) (f a)
triBLens :: Lens Triangle Point
triBLens f (Triangle a b c) = fmap (\b' -> Triangle a b' c) (f b)
triCLens :: Lens Triangle Point
triCLens f (Triangle a b c) = fmap (\c' -> Triangle a b c') (f c)
