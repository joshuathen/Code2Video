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
        # Define the lecture lines for the intro
        lecture_lines = [
            'Meet Vector Vic, a robot on a mission.', 
            'He has two movement paths, V1 and V2.', 
            'His goal is to reach the target vector B.', 
            'How many steps of each path does he take?', 
            'We model this quest as a linear system.'
        ]
        self.setup_layout("Introduction: The Quest of Vector Vic", lecture_lines)
        
        # Define a coordinate system inside the grid area C2 to F6
        # C2-F6 provides clearance from lecture notes (left) and equation (top)
        plane = NumberPlane(
            x_range=[0, 8, 2],
            y_range=[0, 12, 2],
            x_length=3.5,
            y_length=4.5,
            background_line_style={"stroke_opacity": 0.2},
            axis_config={"include_tip": True}
        )
        self.place_in_area(plane, 'C2', 'F6')

        # Define base colors
        v1_color = "#0000FF" # Blue
        v2_color = "#FF0000" # Red
        b_color = "#00FF00"  # Green
        vic_color = "#FFFFFF" # White
        unknown_color = "#FFFF00" # Yellow

        # === Animation for Lecture Line 1 ===
        # "Meet Vector Vic, a robot on a mission."
        self.lecture[0].set_color(WHITE)
        
        # Vector Vic (robot) at origin
        vic_circle = Circle(radius=0.15, color=vic_color, fill_opacity=1)
        vic_label = Text("V", font_size=16, color=BLACK).move_to(vic_circle.get_center())
        vic_group = VGroup(vic_circle, vic_label)
        vic_group.move_to(plane.c2p(0, 0))
        
        self.play(FadeIn(vic_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "He has two movement paths, V1 and V2."
        self.lecture[1].set_color(WHITE)
        
        v1_arrow = Arrow(plane.c2p(0,0), plane.c2p(2, 1), buff=0, color=v1_color)
        v1_label = Text("V1", font_size=20, color=v1_color).next_to(v1_arrow.get_end(), RIGHT, buff=0.1)
        
        v2_arrow = Arrow(plane.c2p(0,0), plane.c2p(1, 3), buff=0, color=v2_color)
        v2_label = Text("V2", font_size=20, color=v2_color).next_to(v2_arrow.get_end(), LEFT, buff=0.1)
        
        self.play(Create(v1_arrow), Write(v1_label))
        self.play(Create(v2_arrow), Write(v2_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "His goal is to reach the target vector B."
        self.lecture[2].set_color(b_color)
        
        # Target vector B pointing to (7, 11)
        b_arrow = Arrow(plane.c2p(0,0), plane.c2p(7, 11), buff=0, color=b_color)
        b_label = Text("B", font_size=24, color=b_color).next_to(b_arrow.get_end(), UP, buff=0.1)
        
        self.play(Create(b_arrow), Write(b_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "How many steps of each path does he take?"
        self.lecture[3].set_color(unknown_color)
        
        # Step-by-step path to B: 2*V1 + 3*V2
        path_v1_1 = Arrow(plane.c2p(0,0), plane.c2p(2,1), buff=0, color=v1_color, stroke_width=3)
        path_v1_2 = Arrow(plane.c2p(2,1), plane.c2p(4,2), buff=0, color=v1_color, stroke_width=3)
        
        path_v2_1 = Arrow(plane.c2p(4,2), plane.c2p(5,5), buff=0, color=v2_color, stroke_width=3)
        path_v2_2 = Arrow(plane.c2p(5,5), plane.c2p(6,8), buff=0, color=v2_color, stroke_width=3)
        path_v2_3 = Arrow(plane.c2p(6,8), plane.c2p(7,11), buff=0, color=v2_color, stroke_width=3)
        
        # Show two V1s tip-to-tail
        self.play(TransformFromCopy(v1_arrow, path_v1_1))
        self.play(TransformFromCopy(path_v1_1, path_v1_2))
        
        # Show three V2s tip-to-tail
        self.play(TransformFromCopy(v2_arrow, path_v2_1))
        self.play(TransformFromCopy(path_v2_1, path_v2_2))
        self.play(TransformFromCopy(path_v2_2, path_v2_3))
        
        # Move Vic along the resulting path to reach B
        self.play(
            vic_group.animate.move_to(plane.c2p(7, 11)),
            run_time=2,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "We model this quest as a linear system."
        self.lecture[4].set_color(WHITE)
        
        # Equation text at the top, centered over the animation area
        equation = Text("x V1 + y V2 = B", font_size=32, t2c={"x": unknown_color, "y": unknown_color})
        self.place_in_area(equation, 'A3', 'A6', scale_factor=0.9)
        
        self.play(Write(equation))
        self.wait(3)
