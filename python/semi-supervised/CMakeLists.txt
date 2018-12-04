cmake_minimum_required(VERSION 3.12)
project(semi_supervised)

set(CMAKE_CXX_STANDARD 14)

add_executable(semi_supervised
        inference/__init__.py
        inference/distributions.py
        inference/variational.py
        layers/__init__.py
        layers/flow.py
        layers/stochastic.py
        models/__init__.py
        models/dgm.py
        models/vae.py
        utils.py)
