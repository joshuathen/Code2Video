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
        # Setup layout with specific title and lecture lines
        self.setup_layout(
            "Prerequisite 1: Conservation Laws",
            [
                "Momentum conservation governs how these blocks bounce.",
                "Energy conservation limits their total possible movement.",
                "Together, these laws define the system's physical state."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Momentum conservation governs how these blocks bounce.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Momentum Formula
        momentum_formula = MathTex("mv + MV = p", color="#00BFFF")
        # Position momentum_formula in area A3-A5 with scale 1.2
        self.place_in_area(momentum_formula, "A3", "A5", scale_factor=1.2)
        
        # Collision Visualization Setup
        wall = Line(UP, DOWN).scale(1.5)
        self.place_at_grid(wall, "D1")
        wall.shift(LEFT * 0.5)
        
        block_m = Square(side_length=0.6, fill_opacity=0.8, color=BLUE).set_stroke(WHITE, 1)
        block_M = Square(side_length=1.0, fill_opacity=0.8, color=RED).set_stroke(WHITE, 1)
        
        self.place_at_grid(block_m, "D2")
        # Move block_M to grid point D3 to avoid overlap with phase space group
        self.place_at_grid(block_M, "D3")
        
        m_label = Text("m", font_size=18).next_to(block_m, UP, buff=0.1)
        M_label = Text("M", font_size=18).next_to(block_M, UP, buff=0.1)
        
        v_arrow = Arrow(block_m.get_center(), block_m.get_center() + LEFT * 1.0, buff=0, color=BLUE)
        V_arrow = Arrow(block_M.get_center(), block_M.get_center() + LEFT * 0.5, buff=0, color=RED)

        self.play(
            Write(momentum_formula),
            Create(wall),
            FadeIn(block_m), FadeIn(block_M),
            Write(m_label), Write(M_label),
            GrowArrow(v_arrow), GrowArrow(V_arrow)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Energy conservation limits their total possible movement.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Energy Formula
        energy_formula = MathTex(r"\frac{1}{2}mv^2 + \frac{1}{2}MV^2 = E", color="#FF4500")
        # Position energy_formula in area B3-B5 with scale 1.2
        self.place_in_area(energy_formula, "B3", "B5", scale_factor=1.2)
        
        self.play(Write(energy_formula))
        
        # Highlight squared terms
        # Indexing for \frac{1}{2}mv^2 + \frac{1}{2}MV^2 = E
        # 0:1/2, 1:m, 2:v, 3:^, 4:2 ... index logic varies. Use manual slices of the submob.
        v_sq = energy_formula[0][4:6] # Approximate slice for v^2
        V_sq = energy_formula[0][11:13] # Approximate slice for V^2
        
        rect1 = SurroundingRectangle(v_sq, color=YELLOW, buff=0.05)
        rect2 = SurroundingRectangle(V_sq, color=YELLOW, buff=0.05)
        
        self.play(Create(rect1), Create(rect2))
        self.wait(0.5)
        
        # Collision simulation (Simplified visual movement)
        self.play(
            block_m.animate.shift(LEFT * 0.4),
            v_arrow.animate.shift(LEFT * 0.4),
            m_label.animate.shift(LEFT * 0.4),
            block_M.animate.shift(LEFT * 0.6),
            V_arrow.animate.shift(LEFT * 0.6),
            M_label.animate.shift(LEFT * 0.6),
            run_time=0.8
        )
        
        # Change arrow sizes to represent velocity change
        new_v_arrow = Arrow(block_m.get_center(), block_m.get_center() + RIGHT * 0.8, buff=0, color=BLUE)
        new_V_arrow = Arrow(block_M.get_center(), block_M.get_center() + LEFT * 0.2, buff=0, color=RED)
        
        self.play(
            ReplacementTransform(v_arrow, new_v_arrow),
            ReplacementTransform(V_arrow, new_V_arrow)
        )
        
        self.play(FadeOut(rect1), FadeOut(rect2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Together, these laws define the system's physical state.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Phase Space Ellipse
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            x_length=3.5,
            y_length=2.5,
            axis_config={"include_tip": True, "font_size": 18}
        )
        v_axis_label = axes.get_x_axis_label("v", edge=RIGHT, direction=RIGHT, buff=0.1)
        V_axis_label = axes.get_y_axis_label("V", edge=UP, direction=UP, buff=0.1)
        
        phase_space_group = VGroup(axes, v_axis_label, V_axis_label)
        self.place_in_area(phase_space_group, "C4", "F6", scale_factor=1.0)
        
        # Ellipse: axes correspond to sqrt(2E/m) and sqrt(2E/M)
        ellipse = Ellipse(width=2.5, height=1.2, color="#FFD700").move_to(axes.c2p(0, 0))
        
        self.play(
            Create(axes),
            Write(v_axis_label),
            Write(V_axis_label)
        )
        self.play(Create(ellipse))
        
        # Final state: highlight formulas and ellipse
        self.play(
            momentum_formula.animate.set_stroke(width=1),
            energy_formula.animate.set_stroke(width=1),
            ellipse.animate.set_stroke(width=4)
        )
        
        self.wait(2)
