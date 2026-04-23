from setuptools import find_packages, setup

package_name = "aic_lewm_policy"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    package_data={
        "aic_lewm_policy.lewm_vendor": ["LICENSE.le-wm"],
    },
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Joseph",
    maintainer_email="joseph@example.com",
    description="AIC qualification policy scaffold for LEWM-based planning",
    license="Apache-2.0 AND MIT",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "aic-lewm-exp=aic_lewm_policy.experiment_harness:main",
            "aic-lewm-offline-check=aic_lewm_policy.offline_policy_check:main",
            "aic-lewm-replay-check=aic_lewm_policy.replay_policy_check:main",
            "aic-lewm-roadmap=aic_lewm_policy.roadmap_tracking:main",
        ],
    },
)
