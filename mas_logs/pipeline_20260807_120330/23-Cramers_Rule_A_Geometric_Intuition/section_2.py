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
        # Setup the scene
        # Section Title and Lecture Lines from storyboard
        lecture_lines = [
            "Determinants represent the signed area in 2D.",
            "Column vectors v1 and v2 form a parallelogram.",
            "Let's visualize this area on the coordinate grid.",
            "Calculations show the area equals the determinant.",
            "This area is our fundamental scaling factor."
        ]
        self.setup_layout("Prerequisite: Determinant as Area", lecture_lines)
        
        # Grid lines for visual context (subtle background)
        grid_lines = VGroup()
        for r in ["A", "B", "C", "D", "E", "F"]:
            grid_lines.add(Line(self.grid[f"{r}1"], self.grid[f"{r}6"], stroke_opacity=0.1, stroke_width=1, color=GRAY))
        for c in ["1", "2", "3", "4", "5", "6"]:
            grid_lines.add(Line(self.grid[f"A{c}"], self.grid[f"F{c}"], stroke_opacity=0.1, stroke_width=1, color=GRAY))
        self.add(grid_lines)

        # === Animation for Lecture Line 1 ===
        # Line: "Determinants represent the signed area in 2D."
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "Column vectors v1 and v2 form a parallelogram."
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        origin = self.grid["E2"]
        v1_end = self.grid["D4"]
        v2_end = self.grid["C3"]
        
        v1 = Arrow(origin, v1_end, buff=0, color="#FFFF00", stroke_width=4)
        v2 = Arrow(origin, v2_end, buff=0, color="#00FF00", stroke_width=4)
        v1_label = MathTex(r"\vec{v}_1", color="#FFFF00")
        v2_label = MathTex(r"\vec{v}_2", color="#00FF00")
        
        # Place labels according to storyboard and grid rules
        self.place_at_grid(v1_label, "D5", scale_factor=0.8)
        self.place_at_grid(v2_label, "C2", scale_factor=0.8)
        
        self.play(GrowArrow(v1), Write(v1_label))
        self.play(GrowArrow(v2), Write(v2_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "Let's visualize this area on the coordinate grid."
        self.play(self.lecture[2].animate.set_color("#33CCFF"))
        
        v12_end = self.grid["B5"]
        
        l1 = Line(v1_end, v12_end, color="#FFFF00", stroke_opacity=0.5)
        l2 = Line(v2_end, v12_end, color="#00FF00", stroke_opacity=0.5)
        
        poly = Polygon(origin, v1_end, v12_end, v2_end, color="#33CCFF", fill_opacity=0.3, stroke_width=0)
        
        self.play(Create(l1), Create(l2))
        self.play(FadeIn(poly))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line: "Calculations show the area equals the determinant."
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        
        area_label = MathTex(r"\text{Area} = \det(A)", color="#FFFFFF")
        # Fix for Issue 24: Reduced scale_factor from 0.8 to 0.7
        self.place_at_grid(area_label, "C4", scale_factor=0.7)
        
        # Pulse animation for the parallelogram
        self.play(
            poly.animate.scale(1.1, about_point=origin), 
            run_time=0.5, 
            rate_func=there_and_back
        )
        self.play(Write(area_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line: "This area is our fundamental scaling factor."
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        calc = MathTex(r"2 \times 2 - 1 \times 1 = 3", color="#FFFFFF")
        # Fix for Issue 23: Move calculation to Row F to avoid cluttering in the visual area
        self.place_in_area(calc, "F4", "F6", scale_factor=0.8)
        
        self.play(Write(calc))
        self.wait(2)
