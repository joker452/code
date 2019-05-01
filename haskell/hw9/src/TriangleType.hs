module TriangleType
  ( Point (..)
  , Triangle (..) ) where

data Point = Point { coordX :: Integer
                   , coordY :: Integer } deriving (Show, Eq)

data Triangle = Triangle { triA :: Point
                         , triB :: Point
                         , triC :: Point } deriving (Show, Eq)
