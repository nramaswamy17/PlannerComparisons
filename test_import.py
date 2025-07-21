import sys
sys.path.insert(0, '.')

print("Testing direct import...")
try:
    from environments.simple_2d import Simple2D
    print("SUCCESS: Simple2D imported!")
    print("Simple2D class:", Simple2D)
except Exception as e:
    print("FAILED:", e)
    import traceback
    traceback.print_exc()
