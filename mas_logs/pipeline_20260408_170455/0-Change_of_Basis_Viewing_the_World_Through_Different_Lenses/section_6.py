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
        # Setup the layout
        lecture_lines = [
            'First identify basis, build matrix P, and multiply.',
            'This logic powers rotations and cameras in computer graphics.',
            'Viewing the world through different lenses changes everything.'
        ]
        self.setup_layout("Visual Summary and Application", lecture_lines)
        
        # Colors for lines
        COLOR_1 = "#FFFF00" # Yellow
        COLOR_2 = "#00FFFF" # Cyan
        COLOR_FINAL = "#FFFFFF" # White
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_1)
        
        # Step text labels
        step1_text = Text("1. Identify Basis", font_size=24, color=COLOR_1)
        self.place_at_grid(step1_text, "A2", scale_factor=0.7)
        
        step2_text = Text("2. Build Matrix P", font_size=24, color=COLOR_1)
        # Resolved Issue 40: Positioning at A4-A6 to avoid cutoff
        self.place_in_area(step2_text, 'A4', 'A6', scale_factor=0.7)
        
        step3_text = Text("3. Multiply to translate", font_size=24, color=COLOR_1)
        # Resolved Issue 40: Positioning at B4-B6 to avoid cutoff
        self.place_in_area(step3_text, 'B4', 'B6', scale_factor=0.7)
        
        # Visual for basis vectors
        v1 = Arrow(start=ORIGIN, end=RIGHT, color=COLOR_1, buff=0)
        v2 = Arrow(start=ORIGIN, end=UP+RIGHT*0.5, color="#FF8800", buff=0)
        basis_group = VGroup(v1, v2)
        self.place_in_area(basis_group, "B2", "D3", scale_factor=0.8)
        
        # Visual for Matrix P
        p_label = Text("P = ", font_size=24, color=COLOR_1)
        m_elements = VGroup(
            Text("1", font_size=24, color=COLOR_1), Text("0.5", font_size=24, color=COLOR_1),
            Text("0", font_size=24, color=COLOR_1), Text("1", font_size=24, color=COLOR_1)
        ).arrange_in_grid(rows=2, cols=2, buff=0.4)
        l_bracket = Text("[", font_size=42, color=COLOR_1).next_to(m_elements, LEFT, buff=0.1)
        r_bracket = Text("]", font_size=42, color=COLOR_1).next_to(m_elements, RIGHT, buff=0.1)
        matrix_p = VGroup(p_label, l_bracket, m_elements, r_bracket).arrange(RIGHT, buff=0.1)
        
        # Resolved Issue 41: Positioning at D4-F6 and scale 0.8
        self.place_in_area(matrix_p, 'D4', 'F6', scale_factor=0.8)
        
        self.play(
            Write(step1_text), 
            Write(step2_text), 
            Write(step3_text), 
            Create(basis_group), 
            Write(matrix_p)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_2)
        
        # Clear previous visuals
        self.play(
            FadeOut(step1_text), 
            FadeOut(step2_text), 
            FadeOut(step3_text), 
            FadeOut(basis_group), 
            FadeOut(matrix_p)
        )
        
        # 2D Square rotating for graphics demo
        square = Square(side_length=1.5, color=COLOR_2, fill_opacity=0.3)
        self.place_in_area(square, "B2", "E5", scale_factor=1.0)
        
        # Labels for the "Graphic Viewport" logic
        viewport_label = Text("Computer Graphics Logic", font_size=20, color=WHITE)
        # Resolved Issue 42: Positioning at A2-A4 for better centering
        self.place_in_area(viewport_label, 'A2', 'A4', scale_factor=0.8)
        
        self.play(Create(square), Write(viewport_label))
        self.play(Rotate(square, angle=2*PI, run_time=3, rate_func=linear))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_FINAL)
        
        # Final Transition: Fade to black logic
        self.play(
            FadeOut(self.lecture), 
            FadeOut(self.title), 
            FadeOut(square), 
            FadeOut(viewport_label)
        )
        
        final_title = Text("Change of Basis: Different Lenses", font_size=42, color=COLOR_FINAL)
        self.play(Write(final_title))
        self.wait(2)
