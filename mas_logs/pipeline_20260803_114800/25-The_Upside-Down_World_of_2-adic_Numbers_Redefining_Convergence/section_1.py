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
        # Initial Setup
        title_str = "The Familiar Path: Standard Convergence"
        lines = [
            "Standard sums converge when terms approach zero.",
            "We measure distance using the Euclidean metric.",
            "Steps shrink until an ant reaches the crumb."
        ]
        self.setup_layout(title_str, lines)
        self.title.set_color("#ADD8E8")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Formula for sum
        formula = MathTex(r"\sum_{n=1}^\infty \frac{1}{2^n} = 1", color="#ADD8E8")
        # Fix for Issue 23: place_in_area 'B2' to 'C5'
        self.place_in_area(formula, 'B2', 'C5', scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Number line
        nl = NumberLine(
            x_range=[0, 1.25, 0.25],
            length=5,
            include_numbers=True,
            color=WHITE,
            include_tip=True
        )
        # Fix for Issue 24: place_in_area 'E1' to 'F6'
        self.place_in_area(nl, 'E1', 'F6', scale_factor=0.8)
        
        self.play(Create(nl))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Ant and Crumb
        # Crumb at position 1 on the number line
        crumb = Star(color="#FFFF00", n=5, inner_radius=0.1, outer_radius=0.2)
        crumb.move_to(nl.n2p(1) + UP * 0.3)
        
        # Ant using SVGMobject - Fix for Issue 19
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg]
        ant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg")
        ant.set_color("#E6E6FA")
        ant.scale(0.2)
        ant.move_to(nl.n2p(0) + UP * 0.3)
        
        self.play(FadeIn(crumb), FadeIn(ant))
        
        # Jump sequence
        # Jump 1: to 1/2
        jump1_target = nl.n2p(0.5) + UP * 0.3
        jump1_arc = ArcBetweenPoints(ant.get_center(), jump1_target, angle=-PI/3)
        # Highlight segment
        segment1 = Line(nl.n2p(0), nl.n2p(0.5), color="#FFFF00", stroke_width=6)
        
        self.play(
            MoveAlongPath(ant, jump1_arc),
            Create(segment1),
            run_time=1.5
        )
        
        # Jump 2: to 3/4
        jump2_target = nl.n2p(0.75) + UP * 0.3
        jump2_arc = ArcBetweenPoints(ant.get_center(), jump2_target, angle=-PI/3)
        segment2 = Line(nl.n2p(0.5), nl.n2p(0.75), color="#FFFF00", stroke_width=6)
        
        self.play(
            MoveAlongPath(ant, jump2_arc),
            Create(segment2),
            run_time=1.2
        )
        
        # Jump 3: to 7/8
        jump3_target = nl.n2p(0.875) + UP * 0.3
        jump3_arc = ArcBetweenPoints(ant.get_center(), jump3_target, angle=-PI/3)
        segment3 = Line(nl.n2p(0.75), nl.n2p(0.875), color="#FFFF00", stroke_width=6)
        
        self.play(
            MoveAlongPath(ant, jump3_arc),
            Create(segment3),
            run_time=0.8
        )
        
        # Finally reach the crumb
        self.play(ant.animate.move_to(nl.n2p(1) + UP * 0.3), run_time=1)
        
        self.wait(2)
        self.lecture[2].set_color(WHITE)
