import os
import site

# List of nvidia libraries TF needs
libs = ['cudnn', 'cublas', 'cuda_nvrtc', 'cuda_runtime', 'cufft', 'curand']
paths = []

print("--- SEARCHING FOR LIBRARIES ---")
for lib in libs:
    try:
        # dynamic import of nvidia packages
        module = __import__(f'nvidia.{lib}', fromlist=['lib'])
        lib_dir = os.path.join(os.path.dirname(module.__file__), 'lib')
        if os.path.exists(lib_dir):
            print(f"Found {lib}: {lib_dir}")
            paths.append(lib_dir)
        else:
            print(f"WARNING: {lib} module found but 'lib' folder missing.")
    except ImportError:
        print(f"MISSING: Could not find package nvidia.{lib}. Did pip install fail?")

print("\n--- RUN THIS COMMAND BELOW ---")
# Join all found paths with colons
path_string = ":".join(paths)
print(f'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path_string}')
