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

class Section4Scene(TeachingScene):
    def construct(self):
        # Fetching titles and lines from storyboard
        title_text = "Visualizing the Composition via Basis Vectors"
        lecture_lines = [
            "Trace i-hat's path through the first transformation.",
            "Follow its journey through the second step.",
            "These final coordinates become our first column.",
            "Repeat this entire process for j-hat.",
            "The combined matrix is now complete."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_I = "#FF0000"  # Red for i-hat
        COLOR_J = "#00FF00"  # Green for j-hat
        
        # Create Coordinate System - Addressing Issue 35 (Positioned in D3-F6)
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": True},
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(plane, 'D3', 'F6', scale_factor=1.0)
        self.add(plane)
        
        # Create Matrix C structure - Addressing Issue 34 (Positioned in B4-C5)
        matrix_label = Text("C = ", font_size=32)
        bracket_l = Text("[", font_size=40)
        bracket_r = Text("]", font_size=40)
        
        entry_11 = Text("?", font_size=32)
        entry_21 = Text("?", font_size=32)
        entry_12 = Text("?", font_size=32)
        entry_22 = Text("?", font_size=32)
        
        col1 = VGroup(entry_11, entry_21).arrange(DOWN, buff=0.3)
        col2 = VGroup(entry_12, entry_22).arrange(DOWN, buff=0.3)
        entries = VGroup(col1, col2).arrange(RIGHT, buff=0.5)
        
        # Basis vector labels above matrix
        label_i = Text("i", slant=ITALIC, color=COLOR_I, font_size=24).next_to(col1, UP, buff=0.2)
        label_j = Text("j", slant=ITALIC, color=COLOR_J, font_size=24).next_to(col2, UP, buff=0.2)
        
        matrix_content = VGroup(matrix_label, bracket_l, entries, bracket_r).arrange(RIGHT, buff=0.1)
        matrix_group = VGroup(matrix_content, label_i, label_j)
        
        self.place_in_area(matrix_group, 'B4', 'C5', scale_factor=0.8)
        self.add(matrix_group)
        
        # Helper for coordinates
        def p(x, y): return plane.coords_to_point(x, y)
        
        # Basis vectors
        vec_i = Vector([1, 0], color=COLOR_I)
        vec_i.put_start_and_end_on(p(0,0), p(1,0))

        # === Animation for Lecture Line 1 ===
        # Trace i-hat through rotation: (1,0) -> (0,1)
        self.play(self.lecture[0].animate.set_color(COLOR_I))
        self.play(Create(vec_i))
        self.wait(0.5)
        
        self.play(
            vec_i.animate.put_start_and_end_on(p(0,0), p(0,1)),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Trace i-hat through shear: (0,1) -> (1,1)
        self.play(self.lecture[1].animate.set_color(COLOR_I))
        
        self.play(
            vec_i.animate.put_start_and_end_on(p(0,0), p(1,1)),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Final coordinates of i-hat become the first column
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        res11 = Text("1", color=COLOR_I, font_size=32).move_to(entry_11)
        res21 = Text("1", color=COLOR_I, font_size=32).move_to(entry_21)
        
        self.play(
            Transform(entry_11, res11),
            Transform(entry_21, res21)
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Trace j-hat through rotation and shear
        self.play(self.lecture[3].animate.set_color(COLOR_J))
        
        vec_j = Vector([0, 1], color=COLOR_J)
        vec_j.put_start_and_end_on(p(0,0), p(0,1))
        
        self.play(Create(vec_j))
        self.wait(0.5)
        
        # Step 4.1: Rotation: (0,1) -> (-1,0)
        self.play(
            vec_j.animate.put_start_and_end_on(p(0,0), p(-1,0)),
            run_time=1.2
        )
        self.wait(0.3)
        
        # Step 4.2: Shear: (-1,0) -> (-1,0) (invariant, shown with a pulse)
        self.play(
            Indicate(vec_j, scale_factor=1.2, color=COLOR_J),
            run_time=1
        )
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Final coordinates of j-hat become the second column
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        res12 = Text("-1", color=COLOR_J, font_size=32).move_to(entry_12)
        res22 = Text("0", color=COLOR_J, font_size=32).move_to(entry_22)
        
        self.play(
            Transform(entry_12, res12),
            Transform(entry_22, res22)
        )
        
        # Highlight the completed matrix
        box = SurroundingRectangle(matrix_content, color=YELLOW, buff=0.1)
        self.play(Create(box))
        self.wait(2)
