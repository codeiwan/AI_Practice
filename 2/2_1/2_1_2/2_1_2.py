x = 2
t = 10
w = 3
b = 1

for epoch in range(2000):

    print(f'epoch = {epoch}')

    y = x*w + 1*b
    print(f' y  = {y:6.3f}')

    E = (y-t)**2/2
    print(f' E  = {E:.7f}')
    if E < 0.0000001:
        break

    yb = y - t
    xb = yb*w
    wb = yb*x
    bb = yb*1
    print(f' xb = {xb:6.3f}, wb = {wb:6.3f}, bb = {bb:6.3f}')

    lr = 0.005
    w = w - lr*wb
    b = b - lr*bb
    print(f' x  = {x:6.3f}, w  = {w:6.3f}, b  = {b:6.3f}')
