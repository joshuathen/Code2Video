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
        lecture_lines = [
            "Think of functions as mappings between two lines.",
            "Moving an input dot shifts the output dot.",
            "Arrows show how points transform from input to output."
        ]
        self.setup_layout("Prerequisite: Functions as Mappings", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create horizontal axes
        # Issue 25: Adjust area to B2-B6 and E2-E6
        input_axis = NumberLine(x_range=[-3, 3, 1], length=4, include_tip=True, color=WHITE)
        output_axis = NumberLine(x_range=[-3, 3, 1], length=4, include_tip=True, color=WHITE)
        
        self.place_in_area(input_axis, 'B2', 'B6')
        self.place_in_area(output_axis, 'E2', 'E6')
        
        input_label = Text("Input", font_size=20, color=WHITE)
        output_label = Text("Output", font_size=20, color=WHITE)
        
        # Issue 23 & 24: Move labels to B1 and E1
        self.place_at_grid(input_label, 'B1', scale_factor=0.8)
        self.place_at_grid(output_label, 'E1', scale_factor=0.8)
        
        self.play(Create(input_axis), Create(output_axis), Write(input_label), Write(output_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        dot_in = Dot(color=WHITE)
        dot_out = Dot(color=WHITE)
        
        # Mapping f(x) = x + 1. Initial x = -2
        dot_in.move_to(input_axis.n2p(-2))
        dot_out.move_to(output_axis.n2p(-1))
        
        self.play(FadeIn(dot_in), FadeIn(dot_out))
        
        # Animation: dot_in moves from x=-2 to x=2
        # dot_out moves according to f(x)=x+1
        val = ValueTracker(-2)
        dot_in.add_updater(lambda d: d.move_to(input_axis.n2p(val.get_value())))
        dot_out.add_updater(lambda d: d.move_to(output_axis.n2p(val.get_value() + 1)))
        
        self.play(val.animate.set_value(2), run_time=3, rate_func=linear)
        dot_in.clear_updaters()
        dot_out.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Create translucent cyan arrows at x = -1, 0, 1
        arrows = VGroup()
        for x_val in [-1, 0, 1]:
            arr = Arrow(
                start=input_axis.n2p(x_val),
                end=output_axis.n2p(x_val + 1),
                color="#00FFFF",
                buff=0,
                stroke_width=2
            ).set_opacity(0.5)
            arrows.add(arr)
            
        self.play(Create(arrows))
        self.wait(2)
