import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'edge_pose_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sathwik Panchangam',
    maintainer_email='sathwik.nagasai@gmail.com',
    description='Edge-based 6D pose tracking and uncertainty quantification pipeline',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = edge_pose_tracker.talker:main',
            'listener = edge_pose_tracker.listener:main',
            'camera_publisher = edge_pose_tracker.camera_publisher:main',
        ],
    },
)
