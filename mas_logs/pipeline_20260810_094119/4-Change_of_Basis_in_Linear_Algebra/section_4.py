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
        lecture_lines = [
            "Identify new and old basis vectors.",
            "Construct the transition matrix P.",
            "Multiply P by new coordinates.",
            "Result is old coordinate vector.",
            "Calculation complete for basis change."
        ]
        self.setup_layout("Step-by-Step Calculation", lecture_lines)
        
        # Colors for lines
        colors = [BLUE, GREEN, YELLOW, ORANGE, PURPLE]

        # Setup math objects
        matrix_p = MathTex(r"P = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}", font_size=36)
        vec_c = MathTex(r"v_C = \begin{pmatrix} 1 \\ 0 \end{pmatrix}", font_size=36)
        res_eq = MathTex(r"P v_C = v_B", font_size=36)
        
        # Load Assets
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg").scale(0.3)
        pencil_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg").scale(0.3)
        note_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/notebook.svg").scale(0.3)

        # Calculation group for placement
        calc_group = VGroup(matrix_p, vec_c, res_eq).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(calc_group, "B4", "E4", scale_factor=0.75)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        self.place_at_grid(calc_icon, "A5")
        self.play(FadeIn(calc_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        self.place_at_grid(pencil_icon, "A6")
        self.play(FadeIn(pencil_icon), FadeIn(matrix_p))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        self.play(FadeIn(vec_c))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(colors[3])
        self.place_at_grid(note_icon, "F5")
        self.play(FadeIn(note_icon), Write(res_eq))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        self.wait(2)
