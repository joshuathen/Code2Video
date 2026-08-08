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
        # Title and Lecture lines from storyboard
        title_text = "The Mystery of the Treasure Map"
        lecture_lines = [
            "Meet Vector-Bot, searching for hidden treasure.",
            "Two paths define the treasure's exact location.",
            "We represent this system as Ax equals b."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Meet Vector-Bot, searching for hidden treasure.
        self.lecture[0].set_color(YELLOW)
        
        # Coordinate system on the right side
        # Use columns 3-6 to respect B021 gap
        # Fix Issue 22: Move axes to C3-F6
        axes = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"include_tip": True, "color": GREY},
            background_line_style={"stroke_opacity": 0.3}
        ).add_coordinates()
        
        self.place_in_area(axes, "C3", "F6", scale_factor=0.6)
        
        # Issue 19: Use Assets
        # Vector-Bot [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        # Treasure chest [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/treasure.svg]
        vector_bot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg").scale(0.3)
        vector_bot.set_color(YELLOW)
        
        treasure = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/treasure.svg").scale(0.3)
        treasure.set_color(GOLD)
        
        # Place them relative to the scaled axes
        vector_bot.move_to(axes.c2p(0, 0))
        treasure.move_to(axes.c2p(1, 2))
        
        self.play(Create(axes), run_time=1.5)
        self.play(FadeIn(vector_bot, shift=UP), FadeIn(treasure, scale=1.5), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Two paths define the treasure's exact location.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        # Equations: 2x + y = 4 (#1E90FF) and x + 2y = 5 (#FFA500)
        # Line1: y = -2x + 4. Line2: y = -0.5x + 2.5.
        line1 = axes.plot(lambda x: -2*x + 4, x_range=[0.5, 2.5], color="#1E90FF")
        line2 = axes.plot(lambda x: -0.5*x + 2.5, x_range=[-1, 4.5], color="#FFA500")
        
        label1 = MathTex("2x + y = 4", color="#1E90FF", font_size=24)
        label2 = MathTex("x + 2y = 5", color="#FFA500", font_size=24)
        
        # Position labels at the bottom edge of the coordinate area (Row F)
        # Maintaining gap from Row F as per belief B005 is hard here given Critic fix for axes
        # Critic moved axes to C3-F6. I'll place labels at bottom of the area.
        self.place_at_grid(label1, "F3", scale_factor=0.8)
        self.place_at_grid(label2, "F5", scale_factor=0.8)
        
        self.play(Create(line1), Write(label1), run_time=1)
        self.play(Create(line2), Write(label2), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We represent this system as Ax equals b.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)
        
        # Matrix Notation: Ax = b
        matrix_eq = MathTex(
            r"\begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}",
            r"\begin{bmatrix} x \\ y \end{bmatrix}",
            "=",
            r"\begin{bmatrix} 4 \\ 5 \end{bmatrix}",
            font_size=34,
            color=WHITE
        )
        # Fix Issue 20: Move matrix_eq to A4-A6
        self.place_in_area(matrix_eq, "A4", "A6", scale_factor=0.7)
        
        ax_b_label = MathTex("Ax = b", color=WHITE, font_size=38)
        # Fix Issue 21: Move ax_b_label to B4-B6
        self.place_in_area(ax_b_label, "B4", "B6", scale_factor=0.7)
        
        # Transformation from lines/labels to matrix notation
        self.play(
            FadeOut(line1), FadeOut(line2),
            ReplacementTransform(label1, matrix_eq),
            ReplacementTransform(label2, matrix_eq),
            run_time=2
        )
        self.play(Write(ax_b_label))
        self.wait(2)
