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
        # Setup lines and title
        lines = [
            "Let's treat polynomials as vectors in a space.",
            "Adding two curves creates a third, combined curve.",
            "This is identical to adding arrows in space.",
            "Scalar multiplication scales the height of the entire curve.",
            "Polynomials satisfy the same logic as geometric arrows."
        ]
        self.setup_layout("Abstract Example 1: The Space of Polynomials", lines)

        # Axis and Plots setup
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-4, 6, 2],
            axis_config={"include_tip": True},
            x_length=5,
            y_length=5
        )
        self.place_in_area(axes, "B2", "E5")
        
        p1_color = "#00FF00"  # Original prompt Green
        p2_color = "#0000FF"  # Original prompt Blue
        p3_color = "#FF0000"  # Sum Red
        scale_color = "#FFFF00" # Scaled Yellow
        
        p1 = axes.plot(lambda x: x**2 + 1, color=p1_color)
        p2 = axes.plot(lambda x: 2*x - 3, color=p2_color)
        
        # Fixed FileNotFoundError by using Text instead of MathTex/Matrix
        p1_label = Text("P1(x) = x^2 + 1", color=p1_color, font_size=24)
        p2_label = Text("P2(x) = 2x - 3", color=p2_color, font_size=24)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(p1_color)
        self.play(Create(axes), Create(p1), Create(p2))
        self.place_at_grid(p1_label, "A2")
        self.place_at_grid(p2_label, "A5")
        self.play(Write(p1_label), Write(p2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(p3_color)
        p3 = axes.plot(lambda x: x**2 + 2*x - 2, color=p3_color)
        p3_label = Text("P3(x) = P1 + P2", color=p3_color, font_size=24)
        self.place_at_grid(p3_label, "F4")
        
        # Visualizing addition at specific points
        addition_lines = VGroup()
        for x_val in [-1, 0, 1]:
            y1 = x_val**2 + 1
            y2 = 2*x_val - 3
            line = DashedLine(
                axes.c2p(x_val, y1), 
                axes.c2p(x_val, y1 + y2), 
                color="#AAAAAA", 
                stroke_width=2
            )
            addition_lines.add(line)
            
        self.play(Create(addition_lines))
        self.play(Create(p3), Write(p3_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        arrow_group = VGroup()
        a1 = Arrow(ORIGIN, RIGHT + UP, buff=0, color=WHITE)
        a2 = Arrow(RIGHT + UP, 2*RIGHT + 0.5*UP, buff=0, color=WHITE)
        a_sum = Arrow(ORIGIN, 2*RIGHT + 0.5*UP, buff=0, color="#FF0000")
        arrow_group.add(a1, a2, a_sum)
        self.place_in_area(arrow_group, "A1", "B2", scale_factor=0.4)
        
        self.play(Create(arrow_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(scale_color)
        p1_scaled = axes.plot(lambda x: 2 * (x**2 + 1), color=scale_color)
        p1_scaled_label = Text("2 * P1(x)", color=scale_color, font_size=24)
        self.place_at_grid(p1_scaled_label, "B5")
        
        self.play(
            Transform(p1.copy(), p1_scaled),
            Write(p1_scaled_label),
            addition_lines.animate.set_opacity(0)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        
        # Vector simulation without using LaTeX Matrix
        v1_entries = VGroup(Text("1", font_size=24), Text("0", font_size=24), Text("1", font_size=24)).arrange(DOWN, buff=0.15)
        v1 = VGroup(Text("[", font_size=40), v1_entries, Text("]", font_size=40)).arrange(RIGHT, buff=0.1).scale(0.6)
        
        v2_entries = VGroup(Text("0", font_size=24), Text("2", font_size=24), Text("-3", font_size=24)).arrange(DOWN, buff=0.15)
        v2 = VGroup(Text("[", font_size=40), v2_entries, Text("]", font_size=40)).arrange(RIGHT, buff=0.1).scale(0.6)
        
        self.place_at_grid(v1, "D2")
        self.place_at_grid(v2, "D5")
        
        # Flashing text
        logic_text = Text("Logic is Identical", color=WHITE, font_size=32)
        self.place_at_grid(logic_text, "C3")
        
        # Fade into container representation
        container = RoundedRectangle(height=3, width=5, corner_radius=0.2, color=GRAY)
        container_text = Text("Vector Space", font_size=18, color=GRAY)
        container_group = VGroup(container, container_text).arrange(UP, buff=0.1)
        self.place_in_area(container_group, "A1", "F6")

        self.play(
            ReplacementTransform(p1_label, v1),
            ReplacementTransform(p2_label, v2),
            FadeOut(axes), FadeOut(p1), FadeOut(p2), FadeOut(p3), 
            FadeOut(p3_label), FadeOut(p1_scaled), FadeOut(p1_scaled_label),
            FadeOut(arrow_group)
        )
        
        self.play(FadeIn(logic_text))
        self.play(Indicate(logic_text))
        self.wait(0.5)
        self.play(FadeOut(logic_text))
        
        self.play(
            FadeIn(container_group),
            v1.animate.move_to(container_group.get_center() + LEFT),
            v2.animate.move_to(container_group.get_center() + RIGHT)
        )
        self.wait(2)
