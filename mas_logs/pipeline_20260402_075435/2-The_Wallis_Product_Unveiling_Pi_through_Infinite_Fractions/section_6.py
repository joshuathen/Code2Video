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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        lecture_lines = [
            "This product links to Stirling's approximation of factorials.",
            "It even appears in the physics of hydrogen atoms.",
            "Simple fractions reveal the universe's hidden mathematical clockwork."
        ]
        self.setup_layout("Application: Why It Matters", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(YELLOW), run_time=0.5)

        # Wallis Product Formula
        wallis_formula = Text(
            "pi/2 = product(4n^2 / (4n^2 - 1))",
            color=WHITE,
            font_size=24
        )
        self.place_in_area(wallis_formula, "B1", "C3", scale_factor=0.6)

        # Stirling's Approximation Formula
        stirling_formula = Text(
            "n! ~ sqrt(2*pi*n) * (n/e)^n",
            color=WHITE,
            font_size=24
        )
        self.place_in_area(stirling_formula, "B4", "C6", scale_factor=0.6)

        # Fade in both formulas in white (#FFFFFF)
        self.play(
            FadeIn(wallis_formula),
            FadeIn(stirling_formula)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2, reset previous
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            run_time=0.5
        )

        # Bohr atom model [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/atom.svg] in blue (#58C4DD)
        atom_model = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/atom.svg")
        atom_model.set_color("#58C4DD")
        self.place_in_area(atom_model, "D1", "F3", scale_factor=1.2)
        
        # Add dynamic rotation for visual interest
        atom_model.add_updater(lambda m, dt: m.rotate(dt * 0.5))

        # Green Bell Curve (#4CAF50) representing probability
        axes = Axes(
            x_range=[-3, 3],
            y_range=[0, 1],
            axis_config={"include_tip": False, "include_ticks": False, "stroke_width": 1},
            tips=False
        ).set_color(GRAY)
        
        bell_curve = axes.plot(
            lambda x: np.exp(-x**2), 
            color="#4CAF50",
            stroke_width=3
        )
        
        curve_group = VGroup(axes, bell_curve)
        self.place_in_area(curve_group, "D4", "F6", scale_factor=0.8)

        # Show atom model and bell curve
        self.play(
            FadeIn(atom_model),
            Create(bell_curve),
            FadeIn(axes)
        )
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3, reset previous
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
            run_time=0.5
        )

        # Formulas and models glow briefly with a white pulse (#FFFFFF)
        all_elements = VGroup(wallis_formula, stirling_formula, atom_model, curve_group)
        
        self.play(
            all_elements.animate.set_color(WHITE).scale(1.1),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(2)

        # Final cleanup: everything fades to black
        self.play(
            FadeOut(all_elements),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
