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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup the scene with title and lecture lines
        self.setup_layout(
            "The Meaning of 'e': Continuous Growth", 
            [
                "The number $e$ represents continuous, natural growth.", 
                "This growth pushes values away from the origin.", 
                "It accelerates straight ahead along the real line."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight line 1 in green and show 'e'
        self.lecture[0].set_color("#00FF00")
        
        # Display the constant 'e' approx 2.718
        e_text = Text("e ≈ 2.718", color="#00FF00")
        self.place_in_area(e_text, "B1", "B6", scale_factor=1.0)
        
        self.play(Write(e_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Vector on the real line growing continuously
        # Represent the real line segment using the grid
        line_start = self.grid["D1"]
        line_end = self.grid["D6"]
        
        real_line = Line(line_start, line_end, color=WHITE, stroke_width=2)
        ticks = VGroup()
        for i in range(6):
            tick_x = line_start[0] + i
            tick = Line([tick_x, line_start[1]-0.1, 0], [tick_x, line_start[1]+0.1, 0], color=WHITE)
            ticks.add(tick)
            
        self.play(Create(real_line), Create(ticks))
        
        # Value tracker for the exponent x (growth = e^x)
        x_tracker = ValueTracker(0) # e^0 = 1
        
        # The vector starts at origin (D1) and its length is e^x
        # We cap growth so it stays within the D1-D6 bounds (5 units)
        growth_vector = Arrow(
            start=line_start,
            end=line_start + RIGHT,
            buff=0,
            color=WHITE,
            stroke_width=5
        )
        
        # Use add_updater for efficiency
        growth_vector.add_updater(
            lambda m: m.put_start_and_end_on(
                line_start, 
                line_start + RIGHT * np.exp(x_tracker.get_value())
            )
        )
        
        self.play(GrowArrow(growth_vector))
        # Initial growth phase
        self.play(x_tracker.animate.set_value(0.5), run_time=1.5, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Label growth 'e^x' and highlight in yellow
        self.lecture[2].set_color("#FFFF00")
        
        e_x_label = Text("e^x", color="#FFFF00")
        # Position label just above the vector head
        e_x_label.add_updater(
            lambda m: m.move_to(growth_vector.get_end() + UP * 0.4)
        )
        
        self.play(
            FadeIn(e_x_label),
            growth_vector.animate.set_color("#FFFF00"),
            run_time=0.5
        )
        
        # Accelerate: x grows from 0.5 up to ~1.6 (exp(1.6) is ~4.95)
        self.play(
            x_tracker.animate.set_value(1.6),
            run_time=2.5,
            rate_func=exponential_speed_up
        )
        self.wait(2)
