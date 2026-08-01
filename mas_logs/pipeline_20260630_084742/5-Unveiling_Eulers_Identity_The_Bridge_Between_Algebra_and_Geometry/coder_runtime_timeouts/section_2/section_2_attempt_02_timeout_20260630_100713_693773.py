from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        # Removing LaTeX $ symbols to avoid rendering issues with Text class
        clean_lines = [line.replace("$", "") for line in lecture_lines]
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in clean_lines]
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
        self.setup_layout(
            "The Meaning of 'e': Continuous Growth", 
            [
                "The number e represents continuous, natural growth.", 
                "This growth pushes values away from the origin.", 
                "It accelerates straight ahead along the real line."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Color: Green (#00FF00) for line and formula
        self.lecture[0].set_color("#00FF00")
        
        # Replaced MathTex with Text to resolve 'latex' FileNotFoundError
        e_val = Text("e ≈ 2.718", color="#00FF00", font_size=32)
        self.place_in_area(e_val, "B2", "B5", scale_factor=1.2)
        
        self.play(Write(e_val))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a vector on the real line growing continuously to the right.
        
        # Number line setup: include_numbers=False to avoid MathTex dependency
        nl = NumberLine(
            x_range=[0, 5, 1],
            length=5,
            include_numbers=False,
            color=WHITE
        )
        self.place_in_area(nl, "D1", "D6")
        
        # Add manual labels using Text to avoid LaTeX
        for x in range(6):
            label = Text(str(x), font_size=16).next_to(nl.n2p(x), DOWN)
            self.add(label)
        
        # Growth tracker for exponent x (e^x)
        x_tracker = ValueTracker(0) # Start at e^0 = 1
        
        # Growth vector setup
        growth_vector = Arrow(
            start=nl.n2p(0),
            end=nl.n2p(1),
            buff=0,
            color=WHITE,
            stroke_width=5
        )
        
        # Vector updater: magnitude follows e^x
        growth_vector.add_updater(
            lambda m: m.put_start_and_end_on(
                nl.n2p(0), 
                nl.n2p(np.exp(x_tracker.get_value()))
            )
        )
        
        self.play(Create(nl))
        self.play(GrowArrow(growth_vector))
        
        # Initial slow growth
        self.play(x_tracker.animate.set_value(0.5), run_time=1.5, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Label the growth 'e^x' and highlight its outward push in yellow (#FFFF00).
        self.lecture[2].set_color("#FFFF00")
        
        # Replaced MathTex with Text to avoid LaTeX dependency
        e_x_label = Text("e^x", color="#FFFF00", font_size=32)
        # Position label relative to the vector tip
        e_x_label.add_updater(
            lambda m: m.move_to(growth_vector.get_end() + UP * 0.5)
        )
        
        # Highlight: Vector turns yellow, label appears
        self.play(
            FadeIn(e_x_label),
            growth_vector.animate.set_color("#FFFF00"),
            run_time=0.8
        )
        
        # Animate acceleration: x goes from 0.5 to 1.6 (e^1.6 ≈ 4.95, fitting nl [0,5])
        self.play(
            x_tracker.animate.set_value(1.6),
            run_time=3.5,
            rate_func=linear
        )
        self.wait(2)