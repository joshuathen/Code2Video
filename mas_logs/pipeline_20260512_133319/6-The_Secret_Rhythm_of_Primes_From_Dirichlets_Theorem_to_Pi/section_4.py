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
        # Initialization
        title_str = "Connecting Discrete to Continuous: The Pi Link"
        lecture_lines = [
            "A unit circle reveals hidden connections to our sequences.",
            "The Leibniz formula alternates sums of odd number reciprocals.",
            "Positive and negative terms follow the prime lane pattern.",
            "Counting lattice points inside the circle links integers to area.",
            "This infinite series eventually converges to Pi over four."
        ]
        self.setup_layout(title_str, lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Asset integration: grid and circle
        grid_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/grid.svg")
        circle_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/circle.svg")
        
        self.place_in_area(grid_svg, "A1", "D6", scale_factor=2.4)
        self.place_in_area(circle_svg, "A1", "D6", scale_factor=2.4)
        grid_svg.set_color(GREY_E)
        circle_svg.set_color(WHITE)
        
        # Lattice points marked in #FFFFFF
        lattice_dots = VGroup()
        center_ref = grid_svg.get_center()
        step_val = 0.4
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                pos = center_ref + np.array([dx * step_val, dy * step_val, 0])
                dot = Dot(point=pos, radius=0.04, color=WHITE)
                lattice_dots.add(dot)

        self.play(FadeIn(grid_svg), Create(circle_svg), FadeIn(lattice_dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        
        # Leibniz formula "1 - 1/3 + 1/5 - 1/7..." in #FFFFFF
        def make_frac(n, d, color=WHITE):
            if d == "1": return Text(n, font_size=24, color=color)
            return VGroup(
                Text(n, font_size=20, color=color),
                Line(LEFT*0.15, RIGHT*0.15, stroke_width=2, color=color),
                Text(d, font_size=20, color=color)
            ).arrange(DOWN, buff=0.05)

        denoms = ["1", "3", "5", "7", "9", "11", "13", "15"]
        formula = VGroup()
        for idx, d_str in enumerate(denoms):
            if idx > 0:
                sign_char = "-" if idx % 2 != 0 else "+"
                formula.add(Text(sign_char, font_size=24, color=WHITE))
            formula.add(make_frac("1", d_str))
        
        formula.add(Text("...", font_size=24, color=WHITE)).arrange(RIGHT, buff=0.2)
        
        # Position formula at F2-F6 (Issue 43)
        self.place_in_area(formula, "F2", "F6", scale_factor=0.7)
        
        # Scroll effect
        f_target = formula.get_center()
        formula.move_to(f_target + RIGHT * 4)
        self.play(formula.animate.move_to(f_target), run_time=2.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF00FF")
        )
        
        # Terms coloring: 4n+1 (#00FFFF), 4n+3 (#FF00FF)
        color_anims = []
        for i in range(len(denoms)):
            val = int(denoms[i])
            c = "#00FFFF" if val % 4 == 1 else "#FF00FF"
            color_anims.append(formula[2*i].animate.set_color(c))
            
        self.play(*color_anims)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#FFFF00")
        )
        
        # Lattice points inside boundary glow #FFFF00
        glow_anims = []
        radius_sq = 1.3**2 # Visual threshold for the lattice/circle area
        for dot in lattice_dots:
            dist_sq = np.sum((dot.get_center() - center_ref)**2)
            if dist_sq < radius_sq:
                glow_anims.append(dot.animate.set_color("#FFFF00").scale(1.5))
        
        if glow_anims:
            self.play(*glow_anims)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#00FF00")
        )
        
        # Convergence label at E5 (Issue 44)
        convergence_msg = Text("Sum ≈ π / 4", font_size=32, color="#00FF00")
        self.place_at_grid(convergence_msg, "E5", scale_factor=0.8)
        
        self.play(Write(convergence_msg))
        self.wait(2)
