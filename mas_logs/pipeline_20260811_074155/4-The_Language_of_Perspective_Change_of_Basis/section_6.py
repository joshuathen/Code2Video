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

class Section6Scene(TeachingScene):
    def construct(self):
        # Fetching data from storyboard
        title = "Summary and Visual Wrap-up"
        lecture_lines = [
            "Change of basis alters descriptions, not the physical space.",
            "Matrix operations allow us to switch between viewpoints easily.",
            "The geometry stays fixed while our perspective shifts."
        ]
        
        self.setup_layout(title, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Lecture: "Change of basis alters descriptions, not the physical space."
        # Visual: Flowchart [v]_B -> [P] -> [v]_Standard in white (#FFFFFF)
        self.lecture[0].set_color(YELLOW)
        
        v_b = MathTex(r"[\vec{v}]_{\mathcal{B}}", color=WHITE)
        p_mat = MathTex(r"P", color=WHITE)
        v_s = MathTex(r"[\vec{v}]_{\mathcal{S}}", color=WHITE)
        arrow1 = Arrow(start=LEFT, end=RIGHT, color=WHITE, buff=0.1)
        arrow2 = Arrow(start=LEFT, end=RIGHT, color=WHITE, buff=0.1)
        
        flowchart = VGroup(v_b, arrow1, p_mat, arrow2, v_s).arrange(RIGHT, buff=0.3)
        # Fix for Issue 34: adjusted area to prevent crowding
        self.place_in_area(flowchart, "A2", "B5", scale_factor=0.9)
        
        self.play(Write(flowchart))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Lecture: "Matrix operations allow us to switch between viewpoints easily."
        # Visual: Morph the background grid smoothly from standard to skewed and back.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Use a small NumberPlane for the demonstration
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3,
            y_length=3,
            background_line_style={"stroke_opacity": 0.4}
        )
        # Fix for Issue 35: adjusted area and scale to prevent crowding
        self.place_in_area(plane, "C2", "E5", scale_factor=0.8)
        
        # A static point to show it doesn't move
        point_coord = plane.c2p(1, 1)
        point = Dot(point_coord, color=RED)
        point_label = MathTex(r"\vec{v}", color=RED).next_to(point, UR, buff=0.1)
        
        self.play(Create(plane), Create(point), Write(point_label))
        self.wait(0.5)
        
        # Skew matrix
        matrix = [[1, 1], [0.5, 1]]
        
        # Animate morphing (skewing)
        self.play(
            plane.animate.apply_matrix(matrix),
            run_time=2
        )
        self.wait(1)
        
        # Animate back to standard
        self.play(
            plane.animate.apply_matrix(np.linalg.inv(matrix)),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Lecture: "The geometry stays fixed while our perspective shifts."
        # Visual: Fade in the text 'Same Space, Different Perspectives' in light blue (#ADD8E6).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        summary_text = Text("Same Space, Different Perspectives", font_size=24, color="#ADD8E6")
        # Fix for Issue 33: moved text to bottom to avoid overlap
        self.place_in_area(summary_text, 'F2', 'F5', scale_factor=1.0)
        
        self.play(FadeIn(summary_text, shift=UP))
        self.wait(3)
