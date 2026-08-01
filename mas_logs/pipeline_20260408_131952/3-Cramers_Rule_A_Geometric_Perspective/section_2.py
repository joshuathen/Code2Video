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
        # Setup the layout with updated teaching script
        self.setup_layout(
            "Prerequisite: The Determinant as Area", 
            [
                "Start with our two direction vectors, V1 and V2.", 
                "They span a parallelogram in the coordinate plane.", 
                "This shape's area is the key to our solution.", 
                "Algebraically, we call this area the determinant.", 
                "For these vectors, the determinant equals five."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        # Blue vector V1 #0000FF and red vector V2 #FF0000
        # Position using grid indices: Origin at F2, V1 at E4, V2 at C3
        v1 = Arrow(self.grid['F2'], self.grid['E4'], buff=0, color="#0000FF", stroke_width=6)
        v2 = Arrow(self.grid['F2'], self.grid['C3'], buff=0, color="#FF0000", stroke_width=6)
        
        # Labels for vectors
        v1_label = Text("V1", font_size=24, color="#0000FF")
        # Issue 31 Fix: v1_label at E6, scale_factor=0.8
        self.place_at_grid(v1_label, 'E6', scale_factor=0.8)
        
        v2_label = Text("V2", font_size=24, color="#FF0000")
        # Issue 32 Fix: v2_label at A3, scale_factor=0.8
        self.place_at_grid(v2_label, 'A3', scale_factor=0.8)

        self.play(Create(v1), Create(v2))
        self.play(Write(v1_label), Write(v2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00") # Yellow highlight for parallelogram concept
        
        # A yellow parallelogram #FFFF00 at 0.3 opacity forms between V1 and V2
        # Vertices: Origin F2, tip V1 E4, sum point B5, tip V2 C3
        poly_points = [self.grid['F2'], self.grid['E4'], self.grid['B5'], self.grid['C3']]
        parallelogram = Polygon(
            *poly_points, 
            fill_color="#FFFF00", 
            fill_opacity=0.3, 
            stroke_width=0
        )
        
        self.play(FadeIn(parallelogram))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        # The area of the parallelogram is labeled 'Area = 5' in yellow #FFFF00.
        area_label = Text("Area = 5", font_size=32, color="#FFFF00")
        self.place_in_area(area_label, 'C4', 'D5') # Center in the parallelogram body
        
        self.play(Write(area_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFFFF")
        
        # The boundary vectors V1 and V2 flash white #FFFFFF simultaneously.
        self.play(
            v1.animate.set_color("#FFFFFF"),
            v2.animate.set_color("#FFFFFF"),
            run_time=0.4
        )
        self.play(
            v1.animate.set_color("#0000FF"),
            v2.animate.set_color("#FF0000"),
            run_time=0.4
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFFFF")
        
        # Text 'det(V1, V2) = 5' appears below the shape in white #FFFFFF.
        det_text = Text("det(V1, V2) = 5", font_size=32, color="#FFFFFF")
        # Issue 33 Fix: det_text in area F3-F5, scale_factor=0.7
        self.place_in_area(det_text, 'F3', 'F5', scale_factor=0.7)
        
        self.play(Write(det_text))
        self.wait(3)
