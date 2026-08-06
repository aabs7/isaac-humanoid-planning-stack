"""The Isaac Sim foundation -- everything that only exists in simulation.

    launch       app launcher + shared CLI parser (safe to import before the app)
    scene        apartment + G1 + sensors, as an InteractiveScene
    locomotion   the pretrained agile-locomotion policy

Apart from ``launch``, these import ``isaaclab`` at module level: import them only
after ``launch()`` has started the app.
"""
