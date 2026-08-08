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

class Section5Scene(TeachingScene):
    def construct(self):
        # Lecture lines from storyboard
        lecture_lines = [
            "- This is the Fundamental Theorem of Calculus.",
            "- To integrate, just find the antiderivative.",
            "- Evaluate it at the two boundaries.",
            "- Subtract the start from the end.",
            "- No more complex Riemann sums needed."
        ]
        
        self.setup_layout("The Fundamental Theorem of Calculus (FTC)", lecture_lines)
        
        # Colors
        GOLD = "#FFD700"
        BLUE = "#87CEEB"
        GREEN = "#90EE90"
        ORANGE = "#FFA500"
        RED = "#FF6347"

        # === Animation for Lecture Line 1 ===
        # "This is the Fundamental Theorem of Calculus."
        self.lecture[0].set_color(GOLD)
        # Display the formal FTC formula: integral from a to b of f(x)dx = F(b) - F(a) in gold (#FFD700).
        ftc_formula = MathTex(
            r"\int_{a}^{b} f(x) \, dx = F(b) - F(a)",
            color=GOLD
        )
        # Issue 41: Position at B2-D5
        self.place_in_area(ftc_formula, 'B2', 'D5', scale_factor=1.1)
        
        self.play(Write(ftc_formula), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "To integrate, just find the antiderivative."
        self.lecture[1].set_color(BLUE)
        # Visual highlight on the formula to represent the concept
        self.play(Indicate(ftc_formula, color=BLUE))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Evaluate it at the two boundaries."
        self.lecture[2].set_color(GREEN)
        
        # Show a vertical axis representing altitude F(x) [Asset: altitude.svg] with points F(a) and F(b).
        # Asset path from storyboard
        altitude_axis = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/altitude.svg")
        altitude_axis.set_color(WHITE)
        self.place_at_grid(altitude_axis, 'E6', scale_factor=0.8)
        
        label_fb = MathTex("F(b)", color=GREEN, font_size=24)
        label_fa = MathTex("F(a)", color=GREEN, font_size=24)
        
        # Position labels one unit away from the anchor (Belief B012)
        self.place_at_grid(label_fb, 'D6', scale_factor=1.0)
        self.place_at_grid(label_fa, 'F6', scale_factor=1.0)
        
        self.play(
            FadeIn(altitude_axis),
            Write(label_fb),
            Write(label_fa)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Subtract the start from the end."
        self.lecture[3].set_color(ORANGE)
        
        # Animate a bracket [Asset: bracket.svg] measuring the distance between F(b) and F(a) on the axis.
        # Asset path from storyboard
        bracket = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bracket.svg")
        bracket.set_color(ORANGE)
        bracket.rotate(-PI/2) # Orient vertically
        self.place_in_area(bracket, 'D6', 'F6', scale_factor=0.7)
        bracket.shift(LEFT * 0.4) # Offset from axis to prevent overlap
        
        diff_text = MathTex("F(b) - F(a)", color=ORANGE, font_size=20)
        self.place_at_grid(diff_text, 'E5', scale_factor=1.0)
        
        self.play(
            FadeIn(bracket),
            FadeIn(diff_text)
        )
        
        # Connect this distance visually to the total shaded area under the curve f(x).
        axes = Axes(
            x_range=[0, 3, 1], y_range=[0, 3, 1],
            x_length=2, y_length=2,
            axis_config={"include_tip": False}
        ).set_color(GRAY)
        # Position axes in the bottom-left area of the right-side grid
        self.place_in_area(axes, 'E1', 'F3', scale_factor=0.8)
        
        curve = axes.plot(lambda x: 0.2*x**2 + 0.5, x_range=[0.5, 2.5], color=GOLD)
        area = axes.get_area(curve, x_range=[0.5, 2.5], color=GOLD, opacity=0.3)
        
        self.play(Create(axes), Create(curve), FadeIn(area))
        self.play(
            Indicate(area, color=ORANGE),
            Indicate(diff_text, color=GOLD)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "No more complex Riemann sums needed."
        self.lecture[4].set_color(RED)
        
        # Fade out a complex Riemann sum grid to leave only the simple F(b) - F(a) calculation.
        riemann_rects = axes.get_riemann_rectangles(curve, x_range=[0.5, 2.5], dx=0.15, fill_opacity=0.4, color=BLUE)
        
        self.play(FadeIn(riemann_rects))
        self.wait(1)
        self.play(FadeOut(riemann_rects))
        
        self.wait(2)
