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

class Section3Scene(TeachingScene):
    def construct(self):
        title_text = "The Solution: The Power Triangle"
        lecture_lines = [
            "Introducing the Triangle of Power for unified math.",
            "Place the Base at the bottom-left corner.",
            "Put the Exponent at the very top.",
            "The Result sits at the bottom-right corner.",
            "One geometric shape stores all three numbers perfectly."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Helper to get center of area for triangle vertices
        def get_area_center(tl, br):
            tl_pos = self.grid[tl]
            br_pos = self.grid[br]
            return np.array([(tl_pos[0] + br_pos[0]) / 2, (tl_pos[1] + br_pos[1]) / 2, 0])

        # === Animation for Lecture Line 1 ===
        # Draw a large equilateral triangle outline in white (#FFFFFF).
        self.lecture[0].set_color("#FFFFFF")
        
        # Updated vertices based on VideoCritic feedback for label positions
        # Exp at B3-C4 center, Base at E1-F2 center, Result at E5-F6 center
        top_v = get_area_center('B3', 'C4')
        bl_v = get_area_center('E1', 'F2')
        br_v = get_area_center('E5', 'F6')
        
        triangle = Polygon(top_v, bl_v, br_v, color=WHITE, stroke_width=4)
        
        self.play(Create(triangle), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Place the text 'Base' and the number 2 at the bottom-left vertex (#FFD700).
        self.lecture[1].set_color("#FFD700")
        
        base_label = Text("Base", font_size=24, color="#FFD700")
        base_val = Text("2", font_size=36, color="#FFD700")
        base_group = VGroup(base_label, base_val).arrange(DOWN, buff=0.1)
        # Fix for Issue 42: Move to E1-F2 area
        self.place_in_area(base_group, 'E1', 'F2', scale_factor=0.8)
        
        self.play(Write(base_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Place the text 'Exponent' and the number 3 at the top vertex (#1E90FF).
        self.lecture[2].set_color("#1E90FF")
        
        exp_label = Text("Exponent", font_size=24, color="#1E90FF")
        exp_val = Text("3", font_size=36, color="#1E90FF")
        exp_group = VGroup(exp_label, exp_val).arrange(DOWN, buff=0.1)
        # Fix for Issue 43: Move to B3-C4 area
        self.place_in_area(exp_group, 'B3', 'C4', scale_factor=0.8)
        
        self.play(Write(exp_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Place the text 'Result' and the number 8 at the bottom-right vertex (#32CD32).
        self.lecture[3].set_color("#32CD32")
        
        res_label = Text("Result", font_size=24, color="#32CD32")
        res_val = Text("8", font_size=36, color="#32CD32")
        res_group = VGroup(res_label, res_val).arrange(DOWN, buff=0.1)
        # Fix for Issue 44: Move to E5-F6 area
        self.place_in_area(res_group, 'E5', 'F6', scale_factor=0.8)
        
        self.play(Write(res_group))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Flash the entire triangle once in gold (#FFD700) to signify unification.
        self.lecture[4].set_color("#FFD700")
        
        self.play(
            Flash(triangle, color="#FFD700", line_length=0.4, flash_radius=1.5, num_lines=30),
            triangle.animate.set_color("#FFD700"),
            run_time=2
        )
        self.wait(2)
