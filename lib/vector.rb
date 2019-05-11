class Vector < Array
  def dot(other)
    self.map.with_index do |item, idx|
      self[idx] * other[idx]
    end.sum
  end

  def sub(other)
    self.zip(other).map { |left, right| left - right }
  end
end
