# Converts position from 1D to 2D representation in (left most col, left most row)
def get_col_row(size: int, pos: int):
    return pos % size, pos // size

# convert top left coordinate to (left row, left col)
def get_2D_col_row(size: int, pos: int):
    top_left_row = pos // size
    top_left_col = pos % size
    return top_left_row, top_left_col

def test_get_2D_col_row(size = 21):
    # print(get_col_row(size,0))
    # print(get_col_row(size,10))
    # print(get_col_row(size,100))
    # print(get_col_row(size,221))
    # print(get_col_row(size,413))
    # print(get_col_row(size, 440))
    assert(get_2D_col_row(size,0) == (0,0))
    assert(get_2D_col_row(size,10) == (0,10))
    assert(get_2D_col_row(size, 413) == (19, 14))
    assert(get_2D_col_row(size, 440) == (20, 20))

def get_to_pos(size: int, pos: int, direction: str):
    col, row = get_col_row(size, pos)
    if direction == "NORTH":
        return pos - size if pos >= size else size ** 2 - size + col
    elif direction == "SOUTH":
        return col if pos + size >= size ** 2 else pos + size
    elif direction == "EAST":
        return pos + 1 if col < size - 1 else row * size
    elif direction == "WEST":
        return pos - 1 if col > 0 else (row + 1) * size - 1

test_get_2D_col_row(21)