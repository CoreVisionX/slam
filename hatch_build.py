import os
import subprocess
import sys
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        print("Running custom build hook: build-cpp")
        
        # Define the build command
        # We use the same command as defined in pixi.toml for the build-cpp task
        clean_cmd = ["rm", "-rf", "build"]
        configure_cmd = ["cmake", "-S", "cpp", "-B", "build", "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"]
        build_cmd = ["cmake", "--build", "build"]

        try:
            print(f"Executing: {' '.join(clean_cmd)}")
            subprocess.check_call(clean_cmd)
            
            print(f"Executing: {' '.join(configure_cmd)}")
            subprocess.check_call(configure_cmd)
            
            print(f"Executing: {' '.join(build_cmd)}")
            subprocess.check_call(build_cmd)
            
        except subprocess.CalledProcessError as e:
            print(f"Error during C++ build: {e}")
            sys.exit(1)
