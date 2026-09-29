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
        self.setup_layout("The Gaussian Core: The Optimal Compromise", [
            "The Gaussian is the optimal compromise.",
            "It minimizes uncertainty in both domains.",
            "Narrowing in time broadens frequency.",
            "Widening in time narrows frequency.",
            "It balances both domains perfectly."
        ])

        colors = ["#FFC300", "#FF5733", "#C70039", "#900C3F", "#581845"]

        # Gaussian helper
        # Defined as mobject that doesn't use always_redraw for expensive things
        # But per instruction 11: Build once, update position/value/geometry in place.
        # Since FunctionGraph needs to be rebuilt, we use an updater.
        axes = Axes(x_range=[-3, 3, 1], y_range=[0, 1.2, 0.5], x_length=4, y_length=2, axis_config={"include_tip": False})
        sigma_tracker = ValueTracker(1.0)
        
        gaussian = always_redraw(lambda: axes.plot(
            lambda t: np.exp(-t**2 / (2 * sigma_tracker.get_value()**2)), 
            color=colors[0]
        ))
        
        # Load Assets
        oscilloscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/oscilloscope.svg")
        metronome = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg")
        
        # Layout according to critic feedback
        self.place_in_area(axes, 'C3', 'E6', scale_factor=0.5)
        self.place_at_grid(oscilloscope, 'B2', scale_factor=0.3)
        self.place_at_grid(metronome, 'F2', scale_factor=0.3)
        
        self.add(axes, gaussian, oscilloscope, metronome)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        self.play(sigma_tracker.animate.set_value(0.5), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(colors[3])
        self.play(sigma_tracker.animate.set_value(2.0), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        self.play(sigma_tracker.animate.set_value(1.0), run_time=2)
        self.wait(2)
