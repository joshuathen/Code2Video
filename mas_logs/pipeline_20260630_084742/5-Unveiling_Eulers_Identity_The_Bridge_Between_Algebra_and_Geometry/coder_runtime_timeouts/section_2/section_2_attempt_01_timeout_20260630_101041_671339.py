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
                "The number e represents continuous, natural growth.", 
                "This growth pushes values away from the origin.", 
                "It accelerates straight ahead along the real line."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight line 1 and show the constant 'e'
        self.lecture[0].set_color("#00FF00")
        
        e_val = Text("e ≈ 2.718", color="#00FF00", font_size=36)
        self.place_in_area(e_val, "B1", "B6", scale_factor=1.0)
        
        self.play(Write(e_val))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a vector on the real line growing continuously to the right.
        
        # Use simple line as NumberLine to avoid overhead
        nl_start = self.grid["D1"]
        nl_end = self.grid["D6"]
        nl_main = Line(nl_start, nl_end, color=WHITE)
        
        # Create tick marks and labels 0 to 5
        ticks = VGroup()
        labels = VGroup()
        for i in range(6):
            pos = nl_start + i * RIGHT
            tick = Line(pos + UP*0.1, pos + DOWN*0.1, color=WHITE)
            label = Text(str(i), font_size=18).next_to(tick, DOWN, buff=0.1)
            ticks.add(tick)
            labels.add(label)
            
        nl_group = VGroup(nl_main, ticks, labels)
        
        # Growth tracker for exponent x (e^x)
        # We'll map values such that x=0 -> start (val=1) and e^1.6 ≈ 5 (end)
        x_tracker = ValueTracker(0)
        
        # Growth vector setup
        growth_vector = Arrow(
            start=nl_start,
            end=nl_start + RIGHT,
            buff=0,
            color=WHITE,
            stroke_width=6
        )
        
        # Vector updater: magnitude follows exp(x)
        # Using simple vector math instead of n2p for performance
        growth_vector.add_updater(
            lambda m: m.put_start_and_end_on(
                nl_start, 
                nl_start + np.exp(x_tracker.get_value()) * RIGHT
            )
        )
        
        self.play(Create(nl_group))
        self.play(GrowArrow(growth_vector))
        
        # Slow initial growth
        self.play(x_tracker.animate.set_value(0.4), run_time=1.5, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Label the growth 'e^x' and highlight acceleration in yellow.
        self.lecture[2].set_color("#FFFF00")
        
        e_x_label = Text("e^x", color="#FFFF00", font_size=32)
        # Position label relative to the vector tip
        e_x_label.add_updater(
            lambda m: m.move_to(growth_vector.get_end() + UP * 0.4)
        )
        
        # Vector turns yellow to match line 3
        self.play(
            FadeIn(e_x_label),
            growth_vector.animate.set_color("#FFFF00"),
            run_time=0.8
        )
        
        # Accelerate growth: x goes from 0.4 to approx 1.6 (e^1.6 ≈ 4.95)
        # Total duration is short to prevent timeout
        self.play(
            x_tracker.animate.set_value(1.6),
            run_time=3.0,
            rate_func=linear
        )
        self.wait(2)
