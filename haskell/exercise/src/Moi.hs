{-# LANGUAGE InstanceSigs #-}
module Moi where

import Control.Applicative(liftA3)
import Control.Monad(replicateM)
import Control.Monad.Trans.State
import System.Random

data Die = DieOne
         | DieTwo
         | DieThree
         | DieFour
         | DieFive
         | DieSix
           deriving (Eq, Show)

intToDie :: Int -> Die
intToDie n =
  case n of
    1 -> DieOne
    2 -> DieTwo
    3 -> DieThree
    4 -> DieFour
    5 -> DieFive
    6 -> DieSix
    -- Use this tactic _extremely_ sparingly.
    x -> error $ "intToDie got non 1-6 integer: " ++ show x

rollDie :: State StdGen Die
rollDie = state $ do
    (n, s) <- randomR (1, 6)
    return (intToDie n, s)

rollsToGetN :: Int -> StdGen -> Int
rollsToGetN n  = go n 0 0
    where go :: Int -> Int -> Int -> StdGen -> Int
          go n' sum count gen
             | sum >= n' = count
             | otherwise = let (die, nextGen) = randomR (1, 6) gen
                           in go n' (sum + die) (count + 1) nextGen

rollsCountLogged :: Int -> StdGen -> (Int, [Die])
rollsCountLogged n = go n 0 0
    where go :: Int -> Int -> Int -> StdGen -> (Int, [Die])
          go n' sum count gen
             | sum >= n' = (count, [])
             | otherwise = let (die, nextGen) = randomR (1, 6) gen
                               (count', dies) = go n' (sum + die) (count + 1) nextGen
                           in  (count', intToDie die: dies)

newtype Moi s a = Moi { runMoi :: s -> (a, s) }

instance Functor (Moi s) where
    fmap :: (a -> b) -> Moi s a -> Moi s b
    fmap f (Moi g) = Moi $ \s -> let (a, s') = g s in (f a, s')

instance Applicative (Moi s) where
    pure :: a -> Moi s a
    pure a = Moi $ \s -> (a, s)

    (<*>) :: Moi s (a -> b) -> Moi s a -> Moi s b
    (Moi f) <*> (Moi g) = Moi $ \s -> let (ab, s') = f s
                                          (a, s'') = g s'
                                        in (ab a, s'')


instance Monad (Moi s) where
    return = pure

    (>>=) :: Moi s a -> (a -> Moi s b) -> Moi s b
    (Moi f) >>= g = Moi $ \s -> let (a, s') = f s in runMoi (g a) s'
