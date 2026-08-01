from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        # Removed LaTeX delimiters ($) from lecture_lines to avoid confusion when rendering as Text
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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup the scene with title and lecture lines
        # Replaced LaTeX '$e$' with plain 'e' to avoid potential LaTeX-related issues in strings
        self.setup_layout(
            "The Meaning of 'e': Continuous Growth", 
            [
                "The number e represents continuous, natural growth.", 
                "Growth pushes a point away from the origin.", 
                "This process accelerates straight ahead along the real line."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Color match: Green (#00FF00)
        self.lecture[0].set_color("#00FF00")
        
        # Fixed FileNotFoundError: Replaced MathTex with Text and used Unicode symbol for approx
        e_val = Text("e ≈ 2.718", color="#00FF00")
        self.place_in_area(e_val, "B2", "B5", scale_factor=1.5)
        
        self.play(Write(e_val))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Growth pushes a point away from the origin.
        # Vector on real line growing right.
        
        # Number line setup
        number_line = NumberLine(
            x_range=[0, 6, 1],
            length=5,
            include_numbers=True, label_constructor=Text,
            label_direction=DOWN,
            color=WHITE
        )
        self.place_in_area(number_line, "D1", "D6")
        
        # Growth vector setup
        time_tracker = ValueTracker(0) # e^0 = 1
        
        growth_vector = Arrow(
            start=number_line.n2p(0),
            end=number_line.n2p(1),
            buff=0,
            color=WHITE,
            stroke_width=6
        )
        
        # Vector updater: magnitude follows e^t
        growth_vector.add_updater(
            lambda m: m.put_start_at(number_line.n2p(0)).put_end_at(
                number_line.n2p(np.exp(time_tracker.get_value()))
            )
        )
        
        self.play(Create(number_line))
        self.play(GrowArrow(growth_vector))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight push in yellow (#FFFF00)
        # Label e^x in yellow
        self.lecture[2].set_color("#FFFF00")
        
        # Fixed FileNotFoundError: Replaced MathTex with Text and used Unicode superscript x
        e_x_label = Text("eˣ", color="#FFFF00")
        # Position label relative to the vector tip
        e_x_label.add_updater(
            lambda m: m.move_to(growth_vector.get_end() + UP * 0.4)
        )
        
        # Transform vector color to yellow to show "outward push"
        self.play(
            FadeIn(e_x_label),
            growth_vector.animate.set_color("#FFFF00"),
            run_time=0.8
        )
        
        # Animate the acceleration
        # e^0=1 to e^1.6 ~ 4.95 (within the 0-6 range)
        self.play(
            time_tracker.animate.set_value(1.6),
            run_time=4,
            rate_func=linear
        )
        self.wait(2)