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
        # Data
        lecture_lines = [
            "Linear Algebra study how space moves and changes.",
            "Matrices transform input vectors into new output vectors.",
            "See the geometry behind the numbers in every equation."
        ]
        self.setup_layout("Synthesis: The Geometric Eye", lecture_lines)

        # Colors
        COLOR_X = "#00FF00"
        COLOR_B = "#FF0000"
        COLOR_GEOMETRIC_EYE = "#00FFFF"
        COLOR_ROBOT = "#FFFFFF"
        COLOR_GRID = "#888888"
        COLOR_AREA = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Linear Algebra study how space moves and changes.
        # Match color: White (default for grids)
        self.lecture[0].set_color(WHITE)
        
        grid_base = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1], 
            background_line_style={"stroke_color": COLOR_GRID, "stroke_opacity": 0.5}
        )
        # Issue 43: self.place_in_area(grid_base, 'B1', 'F6', scale_factor=0.8)
        self.place_in_area(grid_base, "B1", "F6", scale_factor=0.8)
        
        # Basis vectors
        i_hat = Vector([1, 0], color=GREEN)
        j_hat = Vector([0, 1], color=RED)
        basis = VGroup(i_hat, j_hat)
        basis.move_to(grid_base.get_center())
        
        # Area highlight
        scaling_area = Rectangle(width=1, height=1, fill_color=COLOR_AREA, fill_opacity=0.3, stroke_width=1)
        scaling_area.move_to(grid_base.get_center() + RIGHT * 0.4 + UP * 0.4) # Slightly shifted for visibility

        self.play(Create(grid_base), run_time=0.8)
        self.play(GrowArrow(i_hat), GrowArrow(j_hat))
        self.play(FadeIn(scaling_area))
        
        # Rapid changes (Montage)
        sheared_grid_points = grid_base.copy().apply_matrix([[1, 1], [0, 1]])
        scaled_grid_points = grid_base.copy().apply_matrix([[0.5, 0], [0, 1.5]])
        
        self.wait(0.3)
        self.play(Transform(grid_base, sheared_grid_points), run_time=0.5)
        self.play(Transform(grid_base, scaled_grid_points), run_time=0.5)
        self.wait(0.5)
        
        self.play(FadeOut(grid_base), FadeOut(basis), FadeOut(scaling_area))

        # === Animation for Lecture Line 2 ===
        # Matrices transform input vectors into new output vectors.
        # Match color: Green (Input Vector x)
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_X)

        equation = MathTex("A", "\\vec{x}", "=", "\\vec{b}", font_size=42, color=WHITE)
        # Issue 41: self.place_in_area(equation, 'A2', 'A5', scale_factor=1.0)
        self.place_in_area(equation, "A2", "A5", scale_factor=1.0)
        
        # Vector x to b
        origin = self.grid["D3"]
        v_x = Vector([1, 1.5], color=COLOR_X)
        v_b = Vector([-1.5, 0.5], color=COLOR_B)
        
        v_x.shift(origin - v_x.get_start())
        v_b.shift(origin - v_b.get_start())
        
        label_x = MathTex("\\vec{x}", color=COLOR_X, font_size=28).next_to(v_x.get_end(), UR, buff=0.1)
        label_b = MathTex("\\vec{b}", color=COLOR_B, font_size=28).next_to(v_b.get_end(), UL, buff=0.1)

        self.play(Write(equation))
        self.play(GrowArrow(v_x), Write(label_x))
        self.wait(0.5)
        self.play(
            Transform(v_x, v_b),
            Transform(label_x, label_b),
            run_time=1.5
        )
        self.wait(1)
        self.play(FadeOut(equation), FadeOut(v_x), FadeOut(label_x))

        # === Animation for Lecture Line 3 ===
        # See the geometry behind the numbers in every equation.
        # Match color: Cyan (Geometric Eye)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_GEOMETRIC_EYE)

        # Issue 26: Use robot SVG asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        try:
            robot_arm = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        except:
            # Fallback if asset not found (though mandated)
            robot_arm = Triangle().scale(0.5) 
            
        robot_arm.set_color(COLOR_ROBOT)
        self.place_at_grid(robot_arm, "D4", scale_factor=1.2)
        
        self.play(FadeIn(robot_arm))
        # Animate movement based on transformation logic (scaling/rotation)
        self.play(
            robot_arm.animate.shift(RIGHT * 0.5).rotate(PI/4),
            run_time=1.0
        )
        self.play(
            robot_arm.animate.shift(LEFT * 1.0).rotate(-PI/2),
            run_time=1.0
        )
        self.wait(0.5)
        
        self.play(FadeOut(robot_arm))
        
        # Geometric Eye text
        eye_text = Text("Geometric Eye", color=COLOR_GEOMETRIC_EYE, font_size=44)
        # Issue 42: self.place_in_area(eye_text, 'D2', 'E5', scale_factor=0.8)
        self.place_in_area(eye_text, "D2", "E5", scale_factor=0.8)
        eye_glow = eye_text.copy().set_stroke(width=8, opacity=0.3)
        
        self.play(Write(eye_text), FadeIn(eye_glow))
        self.play(Indicate(eye_text, color=COLOR_GEOMETRIC_EYE))
        
        self.wait(2)
        
        # Reset color
        self.lecture[2].set_color(WHITE)
