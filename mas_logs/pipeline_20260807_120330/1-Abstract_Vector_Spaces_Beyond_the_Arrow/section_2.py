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
        self.setup_layout(
            "The 'What If' Moment: Generalization", 
            [
                "Can other objects behave like these arrows?", 
                "Consider functions that we add and scale.", 
                "Vector spaces are broader than we often imagine."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Transition from the grid to a black background.
        # We simulate the previous state (grid and arrows) and fade them out.
        sim_plane = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1], 
            background_line_style={"stroke_opacity": 0.4}
        ).scale(0.5).move_to(self.grid["C3"])
        sim_vector = Arrow(sim_plane.c2p(0,0), sim_plane.c2p(1,1), buff=0, color=YELLOW)
        
        self.lecture[0].set_color(YELLOW)
        self.add(sim_plane, sim_vector)
        self.wait(1)
        self.play(FadeOut(sim_plane), FadeOut(sim_vector), run_time=1)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Icons for a 2x2 matrix, a function curve, and a sequence appear. #F08080
        # Assets used per Issue 26.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#F08080")
        
        # 1. 2x2 Matrix Asset
        matrix_mob = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg").set_color("#F08080")
        self.place_at_grid(matrix_mob, "C2", scale_factor=0.8)
        
        # 2. Function curve Asset
        curve_mob = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/curve.svg").set_color("#F08080")
        self.place_at_grid(curve_mob, "C5", scale_factor=0.8)
        
        # 3. Sequence
        sequence_mob = MathTex(r"\{a_n\}_{n=1}^\infty = (a_1, a_2, \dots)", color="#F08080")
        # Positioned at D2-D5 to be adjacent to C row (Issue 30 fix + B005 shift)
        self.place_in_area(sequence_mob, "D2", "D5", scale_factor=0.8)

        self.play(
            FadeIn(matrix_mob),
            FadeIn(curve_mob),
            FadeIn(sequence_mob),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A white boundary #FFFFFF surrounds the icons, labeled as a 'Vector Space V'.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFFFF")
        
        icons_group = VGroup(matrix_mob, curve_mob, sequence_mob)
        boundary = SurroundingRectangle(icons_group, color=WHITE, buff=0.4)
        
        # Label "Vector Space V" centered using place_in_area per B012
        # Positioned at B2-B5 (Issue 31 fix + B005 shift)
        v_label = Text("Vector Space V", font_size=24, color=WHITE)
        self.place_in_area(v_label, "B2", "B5", scale_factor=0.8)
        
        self.play(Create(boundary), run_time=1)
        self.play(Write(v_label), run_time=1)
        self.wait(2)
