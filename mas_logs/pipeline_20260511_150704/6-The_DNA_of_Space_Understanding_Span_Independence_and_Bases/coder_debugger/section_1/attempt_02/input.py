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
        # Initialize Scene
        lecture_lines = [
            'Meet our drone. It moves along two specific vectors.', 
            'We can scale these vectors by any amount.', 
            'Adding these scaled vectors creates a "linear combination."', 
            'By stretching and tip-to-tail placement, we reach targets.', 
            'This "recipe" defines every position the drone can hit.'
        ]
        self.setup_layout("Prerequisites & Linear Combinations", lecture_lines)

        # Coordinate System Setup
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"stroke_width": 2}
        )
        # Position plane in the 6x6 grid area
        self.place_in_area(plane, 'A1', 'F6', scale_factor=0.8)
        origin = plane.c2p(0, 0, 0)

        # Vector Definitions
        vec_a_color = "#FFD700" # Gold
        vec_b_color = "#87CEEB" # Sky Blue
        scaled_a_color = "#FFFF00" # Yellow
        scaled_b_color = "#FF4500" # Orange-Red
        resultant_color = "#FFFFFF" # White

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(vec_a_color)
        
        vec_a = Arrow(start=origin, end=plane.c2p(1, 1, 0), buff=0, color=vec_a_color)
        vec_b = Arrow(start=origin, end=plane.c2p(1, 0, 0), buff=0, color=vec_b_color)
        
        # Fixed: Replaced MathTex with Text to avoid LaTeX dependency error
        label_a = Text("A", color=vec_a_color, font_size=24)
        label_b = Text("B", color=vec_b_color, font_size=24)
        
        # Position labels near vector tips
        label_a.next_to(vec_a.get_end(), UR, buff=0.1)
        label_b.next_to(vec_b.get_end(), DR, buff=0.1)

        self.play(Create(plane), run_time=1)
        self.play(
            GrowArrow(vec_a), 
            GrowArrow(vec_b), 
            Write(label_a), 
            Write(label_b), 
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(scaled_a_color)
        
        # Scale A to 2A
        vec_a_target = Arrow(start=origin, end=plane.c2p(2, 2, 0), buff=0, color=scaled_a_color)
        # Fixed: Replaced MathTex with Text
        label_2a = Text("2A", color=scaled_a_color, font_size=24)
        label_2a.next_to(vec_a_target.get_end(), UR, buff=0.1)

        self.play(
            Transform(vec_a, vec_a_target),
            Transform(label_a, label_2a),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(scaled_b_color)
        
        # Scale B to -1B
        vec_b_target = Arrow(start=origin, end=plane.c2p(-1, 0, 0), buff=0, color=scaled_b_color)
        # Fixed: Replaced MathTex with Text
        label_nb = Text("-1B", color=scaled_b_color, font_size=24)
        label_nb.next_to(vec_b_target.get_end(), DL, buff=0.1)

        self.play(
            Transform(vec_b, vec_b_target),
            Transform(label_b, label_nb),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Shift B tip-to-tail
        self.lecture[3].set_color(vec_b_color)
        
        # Move scaled B to end of scaled A
        target_pos = vec_a.get_end()
        shift_vector = target_pos - vec_b.get_start()
        
        self.play(
            vec_b.animate.shift(shift_vector),
            label_b.animate.shift(shift_vector),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(resultant_color)
        
        # Final resultant vector
        vec_c = Arrow(start=origin, end=vec_b.get_end(), buff=0, color=resultant_color)
        # Fixed: Replaced MathTex with Text
        label_c = Text("2A - B", color=resultant_color, font_size=24)
        label_c.next_to(vec_c.get_end(), UP, buff=0.1)

        self.play(
            Create(vec_c),
            Write(label_c),
            run_time=1.5
        )
        self.wait(2)