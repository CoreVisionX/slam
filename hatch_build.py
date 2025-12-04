import os
import subprocess
import sys
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface


class CustomMetadataHook(MetadataHookInterface):
    """
    Custom metadata hook that appends SLAM_BUILD_VARIANT env var as local version identifier.
    
    Example:
        SLAM_BUILD_VARIANT=jetson -> 0.1.0+jetson
        SLAM_BUILD_VARIANT=rpi    -> 0.1.0+rpi
        (not set)                 -> 0.1.0 (unchanged)
    """
    PLUGIN_NAME = "custom"
    
    def update(self, metadata):
        build_variant = os.environ.get("SLAM_BUILD_VARIANT")
        if build_variant:
            base_version = metadata.get("version", "0.0.0")
            # Strip any existing local version first
            base_version = base_version.split("+")[0]
            new_version = f"{base_version}+{build_variant}"
            metadata["version"] = new_version
            print(f"Build variant '{build_variant}' -> version {new_version}")


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"
    
    def initialize(self, version, build_data):
        print("Running custom build hook: build-cpp")
        
        # Mark as platform-specific wheel (not pure Python)
        build_data['pure_python'] = False
        build_data['infer_tag'] = True
        
        # Define the build command
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
