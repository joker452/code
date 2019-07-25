module MyEnumerable
  def each_until_nil
    niled = false
    self.each {|item|
      if item.nil?
        if niled == false
          niled = true
        end
      end;
      if niled == false
        yield item
      end
    }
  end

  def first_nil_index
    i = 0
    self.each_until_nil {|x| i = i+ 1}
  end
end

class LifeForm
end

class Plant < LifeForm
  def speak
    ""
  end
  
  def num_legs
    0
  end

  def is_name_of_a_data_structure
    false
  end
end

class Animal < LifeForm
  def is_name_of_a_data_structure
    false
  end
end

class Dog < Animal
  def speak
    "bark"
  end

  def num_legs
    4
  end
end

class Giraffe < Animal
  def speak
    ""
  end

  def num_legs
    4
  end
end

class Centipede < Animal
  def speak
    "munch"
  end

  def num_legs
    100
  end
end

class Snake < Animal
  def speak
    "hiss"
  end

  def num_legs
    0
  end
end

class Tree < Plant
  def is_name_of_a_data_structure
    true
  end
end

class Flower < Plant
end

class Vine < Plant
end

#9.
# (a) {x: int, y: {a: int, b: int}, z: int}, permutation types (5 in total), {x: int, y: {a: int, b: int}, z: int, f1: t1, f2: t2,..., fn: tn}
# (b) infinite
# (c) all its subtypes, i.e., all the functions with both argument types in (no argument, {x: int}, {y: int}, {z: int}... 15 in total) and
# return type in (int, {f1: int , f2: t2}, {f1: t1, f2: int}, {f1: int, f2: t2, f3: t3} ...)
# (d) infinite
# (e) all ittes subtypes, i.e., all the functions with both argument types in (no argument, int) and return type in (a)
# (g) infinite
