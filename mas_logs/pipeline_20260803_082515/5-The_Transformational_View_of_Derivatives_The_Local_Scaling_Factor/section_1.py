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

class Section1Scene(TeachingScene):
    def construct(self):
        # Section data
        title = "Introduction: The Number Line Stretcher"
        lines = [
            "Forget graphs for a moment.",
            "Imagine a function as a mapping between number lines.",
            "It morphs the input space into the output space.",
            "Like a rubber band being pulled and compressed.",
            "Points land on new positions after the transformation."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_X = "#FFFFFF"
        COLOR_Y = "#FFFFFF"
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_ARROW = "#44AAFF"

        # === Animation for Lecture Line 1 ===
        # "Forget graphs for a moment."
        self.lecture[0].set_color(YELLOW)
        
        # Create Input line (x) and Output line (y)
        input_line = Line(self.grid["B1"], self.grid["B6"], color=COLOR_X)
        output_line = Line(self.grid["E1"], self.grid["E6"], color=COLOR_Y)
        
        input_label = Text("Input x", font_size=20, color=COLOR_X)
        self.place_in_area(input_label, "A2", "A5", scale_factor=0.8)
        
        output_label = Text("Output y", font_size=20, color=COLOR_Y)
        self.place_in_area(output_label, "D2", "D5", scale_factor=0.8)
        
        self.play(
            Create(input_line), 
            Create(output_line), 
            Write(input_label), 
            Write(output_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Imagine a function as a mapping between number lines."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Add ticks for visualization
        input_ticks = VGroup(*[
            Line(UP*0.1, DOWN*0.1, color=COLOR_X).move_to(self.grid[f"B{i}"]) 
            for i in range(1, 7)
        ])
        output_ticks = VGroup(*[
            Line(UP*0.1, DOWN*0.1, color=COLOR_Y).move_to(self.grid[f"E{i}"]) 
            for i in range(1, 7)
        ])
        
        self.play(Create(input_ticks), Create(output_ticks), run_time=1)
        
        # Setup dot 'x' tracker (0 to 5 represents grid columns 1 to 6)
        x_tracker = ValueTracker(0)
        
        def get_x_pos():
            val = x_tracker.get_value()
            return self.grid["B1"] + (self.grid["B6"] - self.grid["B1"]) * (val / 5)

        dot_x = Dot(color=COLOR_X).add_updater(lambda d: d.move_to(get_x_pos()))
        dot_x_label = MathTex("x", font_size=24, color=COLOR_X)
        dot_x_label.add_updater(lambda m: m.next_to(dot_x, UP, buff=0.1))
        
        self.add(dot_x, dot_x_label)
        self.play(x_tracker.animate.set_value(5), run_time=3, rate_func=linear)
        self.play(x_tracker.animate.set_value(2), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "It morphs the input space into the output space."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Mapping function f(v) where v is in [0, 5]
        # Cubic function showing compression in center and stretching at ends
        def f(v):
            return 2.5 + (v - 2.5)**3 / 6.25

        def get_y_pos():
            val = x_tracker.get_value()
            y_val = f(val)
            return self.grid["E1"] + (self.grid["E6"] - self.grid["E1"]) * (y_val / 5)

        dot_y = Dot(color=COLOR_Y).add_updater(lambda d: d.move_to(get_y_pos()))
        dot_y_label = MathTex("y", font_size=24, color=COLOR_Y)
        dot_y_label.add_updater(lambda m: m.next_to(dot_y, DOWN, buff=0.1))
        
        # Persistent mapping arrow
        mapping_arrow = Arrow(
            start=dot_x.get_center(), 
            end=dot_y.get_center(), 
            buff=0, 
            color=COLOR_ARROW, 
            stroke_width=2
        )
        mapping_arrow.add_updater(lambda a: a.put_start_and_end_on(dot_x.get_center(), dot_y.get_center()))
        
        self.add(dot_y, dot_y_label, mapping_arrow)
        self.play(x_tracker.animate.set_value(0), run_time=1)
        self.play(x_tracker.animate.set_value(5), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Like a rubber band being pulled and compressed."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Show multiple sample points to visualize the morphing
        num_dots = 11
        x_points = [i * 5 / (num_dots - 1) for i in range(num_dots)]
        
        x_dots_samples = VGroup(*[
            Dot(radius=0.04, color=COLOR_X).move_to(
                self.grid["B1"] + (self.grid["B6"] - self.grid["B1"]) * (p / 5)
            ) for p in x_points
        ])
        
        y_dots_samples = VGroup(*[
            Dot(radius=0.04, color=COLOR_Y).move_to(
                self.grid["E1"] + (self.grid["E6"] - self.grid["E1"]) * (f(p) / 5)
            ) for p in x_points
        ])
        
        connections = VGroup(*[
            Line(
                x_dots_samples[i].get_center(), 
                y_dots_samples[i].get_center(), 
                stroke_width=1, 
                stroke_opacity=0.3, 
                color=BLUE_E
            )
            for i in range(num_dots)
        ])
        
        self.play(Create(x_dots_samples), Create(connections), run_time=1)
        self.play(ReplacementTransform(x_dots_samples.copy(), y_dots_samples), run_time=2.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Points land on new positions after the transformation."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Highlight non-uniform segments to emphasize transformation
        # Define a segment [2, 3] on Input Line
        x_seg_start = self.grid["B1"] + (self.grid["B6"] - self.grid["B1"]) * (2/5)
        x_seg_end = self.grid["B1"] + (self.grid["B6"] - self.grid["B1"]) * (3/5)
        x_seg = Line(x_seg_start, x_seg_end, color=COLOR_HIGHLIGHT, stroke_width=8)
        
        # Its mapping on Output Line
        y_seg_start = self.grid["E1"] + (self.grid["E6"] - self.grid["E1"]) * (f(2)/5)
        y_seg_end = self.grid["E1"] + (self.grid["E6"] - self.grid["E1"]) * (f(3)/5)
        y_seg = Line(y_seg_start, y_seg_end, color=COLOR_HIGHLIGHT, stroke_width=8)
        
        self.play(Create(x_seg))
        self.play(TransformFromCopy(x_seg, y_seg), run_time=2)
        
        # Clear focal point movement
        self.play(x_tracker.animate.set_value(2), run_time=1.5)
        self.play(x_tracker.animate.set_value(3), run_time=1.5)
        
        self.wait(2)
