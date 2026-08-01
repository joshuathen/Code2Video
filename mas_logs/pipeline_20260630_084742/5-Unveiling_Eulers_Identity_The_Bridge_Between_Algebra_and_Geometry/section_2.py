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
        # Initializing the layout
        self.setup_layout(
            "The Meaning of 'e': Continuous Growth", 
            [
                "The number $e$ represents continuous, natural growth.", 
                "This growth pushes values away from the origin.", 
                "It accelerates straight ahead along the real line."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line in Green
        self.play(self.lecture[0].animate.set_color("#00FF00"), run_time=0.5)
        
        # Display the constant 'e' in green
        e_val_text = Text("e ≈ 2.718", color="#00FF00")
        # Issue 34: Move to Row A (A2-A5), scale 0.9
        self.place_in_area(e_val_text, "A2", "A5", scale_factor=0.9)
        
        self.play(Write(e_val_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line in White
        self.play(self.lecture[1].animate.set_color(WHITE), run_time=0.5)
        
        # Show a real line and the origin
        # Issue 35: Set number line to D1-D5 to keep margin on right
        origin_pt = self.grid["D1"]
        end_pt = self.grid["D5"]
        number_line = Line(origin_pt, end_pt, color=WHITE, stroke_width=2)
        self.place_in_area(number_line, 'D1', 'D5', scale_factor=1.0)
        
        origin_marker = Dot(origin_pt, color=WHITE, radius=0.08)
        
        # Issue 36: Position origin label at E1, scale 0.8
        origin_label = Text("0", font_size=18)
        self.place_at_grid(origin_label, 'E1', scale_factor=0.8)
        
        # Tracker for the growth vector length
        v_length = ValueTracker(0.01)
        
        # Vector on the real line
        growth_vec = Arrow(
            start=origin_pt,
            end=origin_pt + RIGHT * 0.1,
            buff=0,
            color=WHITE,
            stroke_width=5,
            max_tip_length_to_length_ratio=0.3
        )
        
        # Simple updater for length
        growth_vec.add_updater(
            lambda m: m.put_start_and_end_on(
                origin_pt, 
                origin_pt + RIGHT * v_length.get_value()
            )
        )
        
        self.play(Create(number_line), FadeIn(origin_marker), FadeIn(origin_label))
        self.add(growth_vec)
        
        # Initial growth from origin
        self.play(v_length.animate.set_value(1.2), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight third line in Yellow
        self.play(self.lecture[2].animate.set_color("#FFFF00"), run_time=0.5)
        
        # Label 'e^x' in yellow
        ex_label = Text("e^x", color="#FFFF00", font_size=24)
        
        # Updater to keep label above the tip of the vector
        ex_label.add_updater(
            lambda m: m.move_to(growth_vec.get_end() + UP * 0.35)
        )
        
        # Highlight push by changing color and moving faster
        self.play(
            FadeIn(ex_label),
            growth_vec.animate.set_color("#FFFF00"),
            run_time=0.5
        )
        
        # Continuous acceleration (exponential feel)
        # Vector expands to cover more of the line (staying within D5 boundary)
        # Distance between D1 and D5 is 4 grid units
        self.play(
            v_length.animate.set_value(3.8),
            run_time=3,
            rate_func=rush_into  # Provides acceleration feel
        )
        
        self.wait(2)
