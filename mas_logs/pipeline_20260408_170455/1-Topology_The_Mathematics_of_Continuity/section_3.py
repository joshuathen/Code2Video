from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup the layout with lecture lines
        lecture_lines = [
            'Mathematicians use the word "homeomorphism" for topological equality.',
            'If shape A can morph into B, they are homeomorphs.',
            'Take this classic example: a coffee mug and a doughnut.',
            "The handle's hole is the only thing that really matters.",
            'To a topologist, they are the same geometric object.'
        ]
        self.setup_layout("Homeomorphism: The Coffee Mug Paradox", lecture_lines)
        
        # Colors
        ORANGE = "#E69F00"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Mathematicians use the word "homeomorphism" for topological equality.
        self.play(self.lecture[0].animate.set_color(ORANGE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # If shape A can morph into B, they are homeomorphs.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(ORANGE)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Take this classic example: a coffee mug and a doughnut.
        # Draw a white 2D outline representing a coffee mug (#FFFFFF) 
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/mug.svg] with a distinct handle.
        mug = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/mug.svg")
        mug.set_color(WHITE_COLOR)
        # Fix Issue 37: scaling and position
        self.place_in_area(mug, 'C2', 'F5', scale_factor=1.0)
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(ORANGE),
            Create(mug)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The handle's hole is the only thing that really matters.
        # Change the handle's stroke color to orange (#E69F00) to highlight its importance.
        # Highlight the handle part (often the last sub-mobject in these SVG icons)
        handle_part = mug[-1] if len(mug) > 1 else mug
        
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(ORANGE),
            handle_part.animate.set_stroke(color=ORANGE, width=8)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # To a topologist, they are the same geometric object.
        # Morph the mug outline into a torus (doughnut) shape (#FFFFFF) 
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/dough.svg], with the handle becoming the central hole.
        # Display the label 'HOMEOMORPHISM' in bold orange text (#E69F00) above the shape.
        
        torus = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/dough.svg")
        torus.set_color(WHITE_COLOR)
        # Fix Issue 38: scaling and position
        self.place_in_area(torus, 'C2', 'F5', scale_factor=1.0)
        
        label = Text("HOMEOMORPHISM", weight=BOLD, color=ORANGE)
        # Fix Issue 36: position
        self.place_in_area(label, 'B2', 'B5', scale_factor=0.8)
        
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(ORANGE),
            ReplacementTransform(mug, torus),
            Write(label),
            run_time=2
        )
        
        # Pulse the central hole of the torus with an orange glow (#E69F00) to show its origin.
        pulse_circle = Circle(radius=0.4, color=ORANGE).move_to(torus.get_center())
        pulse_circle.set_stroke(width=12, opacity=0.6)
        self.add(pulse_circle)
        self.play(
            pulse_circle.animate.scale(2.0).set_stroke(opacity=0),
            run_time=1.5,
            rate_func=linear
        )
        self.remove(pulse_circle)
        
        self.wait(3)
