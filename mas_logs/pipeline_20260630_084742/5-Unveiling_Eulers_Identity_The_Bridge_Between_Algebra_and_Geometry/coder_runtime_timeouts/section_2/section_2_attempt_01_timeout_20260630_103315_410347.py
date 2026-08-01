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
        # Setup layout with specific lecture lines
        self.setup_layout(
            "The Meaning of 'e': Continuous Growth", 
            [
                "The number e represents continuous, natural growth.", 
                "This growth pushes values away from the origin.", 
                "It accelerates straight ahead along the real line."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line in Green
        self.play(self.lecture[0].animate.set_color("#00FF00"), run_time=0.5)
        
        # Display the constant 'e' in green
        e_text = Text("e ≈ 2.718", color="#00FF00")
        self.place_in_area(e_text, "B2", "B5", scale_factor=1.2)
        
        self.play(Write(e_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line in White (default is white, but ensure visibility)
        self.play(self.lecture[1].animate.set_color(WHITE), run_time=0.5)
        
        # Show a real line and the origin
        origin_pos = self.grid["D1"]
        end_pos = self.grid["D6"]
        real_line = Line(origin_pos, end_pos, color=WHITE)
        origin_dot = Dot(origin_pos, color=WHITE)
        
        # Value tracker for the length of the growth vector
        # We start at length 1.0 (representing e^0)
        growth_val = ValueTracker(1.0)
        
        # Vector on the real line
        growth_vector = Arrow(
            start=origin_pos,
            end=origin_pos + RIGHT * 1.0,
            buff=0,
            color=WHITE,
            stroke_width=6
        )
        
        # Updater for smooth growth
        growth_vector.add_updater(
            lambda m: m.put_start_and_end_on(
                origin_pos, 
                origin_pos + RIGHT * growth_val.get_value()
            )
        )
        
        self.play(Create(real_line), FadeIn(origin_dot))
        self.play(GrowArrow(growth_vector))
        
        # Initial slow growth
        self.play(growth_val.animate.set_value(1.5), run_time=1.5, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight third line in Yellow
        self.play(self.lecture[2].animate.set_color("#FFFF00"), run_time=0.5)
        
        # Label 'e^x' in yellow
        # Position label relative to the arrow head
        ex_label = Text("e^x", color="#FFFF00")
        ex_label.add_updater(
            lambda m: m.move_to(growth_vector.get_end() + UP * 0.4)
        )
        
        # Highlight the outward push by changing vector color to yellow
        self.play(
            FadeIn(ex_label),
            growth_vector.animate.set_color("#FFFF00"),
            run_time=0.5
        )
        
        # Continuous acceleration (exponential feel)
        # Move growth_val from 1.5 up to 4.5 (near end of grid D6)
        self.play(
            growth_val.animate.set_value(4.5),
            run_time=3,
            rate_func=bezier([0, 0, 1, 1]) # Smooth acceleration
        )
        
        self.wait(2)
