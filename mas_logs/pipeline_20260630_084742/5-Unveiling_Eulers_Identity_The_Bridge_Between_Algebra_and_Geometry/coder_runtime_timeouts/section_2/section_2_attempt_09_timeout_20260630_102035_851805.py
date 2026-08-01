from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets)
        # Ensure self.lecture is a VGroup of Text objects, not the raw strings
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
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
        # Highlight line 1 in green and show 'e'
        # self.lecture is a VGroup, so self.lecture[0] is a Text object
        self.lecture[0].set_color("#00FF00")
        
        # Using Text to avoid dependency on system LaTeX installation
        e_val = Text("e ≈ 2.718", color="#00FF00")
        self.place_in_area(e_val, "B1", "B6", scale_factor=1.2)
        
        self.play(Write(e_val))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Simple Number Line representation
        nl_line = Line(self.grid["D1"], self.grid["D6"], color=WHITE)
        ticks = VGroup()
        for i in range(6):
            tick_pos = self.grid["D1"] + i * RIGHT
            tick = Line(tick_pos + UP*0.1, tick_pos + DOWN*0.1, color=WHITE)
            ticks.add(tick)
            
        nl_group = VGroup(nl_line, ticks)
        self.play(Create(nl_group))

        # Growth tracker for the exponent x (e^x)
        x_tracker = ValueTracker(0)
        
        growth_vector = Arrow(
            start=self.grid["D1"],
            end=self.grid["D1"] + RIGHT,
            buff=0,
            color=WHITE,
            stroke_width=6
        )
        
        # Update vector end based on exp(x)
        growth_vector.add_updater(
            lambda m: m.put_start_and_end_on(
                self.grid["D1"], 
                self.grid["D1"] + np.exp(x_tracker.get_value()) * RIGHT
            )
        )
        
        self.play(GrowArrow(growth_vector))
        # Initial slow growth
        self.play(x_tracker.animate.set_value(0.4), run_time=1.5, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Label 'e^x' and color vector yellow for acceleration.
        self.lecture[2].set_color("#FFFF00")
        
        e_x_label = Text("e^x", color="#FFFF00")
        # Position label relative to the growing vector
        e_x_label.add_updater(
            lambda m: m.move_to(growth_vector.get_end() + UP * 0.4)
        )
        
        self.play(
            FadeIn(e_x_label),
            growth_vector.animate.set_color("#FFFF00"),
            run_time=0.5
        )
        
        # Accelerate: x from 0.4 up to approx 1.6 (e^1.6 is approx 4.95)
        self.play(
            x_tracker.animate.set_value(1.6),
            run_time=3.0,
            rate_func=exponential_speed_up
        )
        self.wait(2)