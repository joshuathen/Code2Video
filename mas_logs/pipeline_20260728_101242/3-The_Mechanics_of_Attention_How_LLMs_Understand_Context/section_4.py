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
        title = "The Math of Matching: Scaled Dot-Product"
        lines = [
            "We multiply Query and Key to find relevance.",
            "This dot-product measures how well they match.",
            "Scaling ensures the mathematical gradients remain stable."
        ]
        self.setup_layout(title, lines)

        # Colors
        Q_COLOR = "#FF0000"
        K_COLOR = "#0000FF"
        HIGHLIGHT_COLOR = "#FFFF00"
        SCORE_BOX_COLOR = WHITE
        HEATMAP_COLORS = ["#FFA500", "#FF8C00", "#FF4500"]

        # === Animation for Lecture Line 1 ===
        # Lecture line color change
        self.play(self.lecture[0].animate.set_color(Q_COLOR))

        # Q vector
        q_vec = Rectangle(height=0.4, width=1.0, color=Q_COLOR, fill_opacity=0.5)
        q_label = Text("Query (Q)", font_size=16, color=Q_COLOR)
        q_group = VGroup(q_vec, q_label).arrange(DOWN, buff=0.1)
        # Fix Issue 29: Move q_group to C2
        self.place_at_grid(q_group, "C2")
        
        # K vectors
        k_vecs = VGroup(*[
            Rectangle(height=0.4, width=1.0, color=K_COLOR, fill_opacity=0.5)
            for _ in range(3)
        ]).arrange(DOWN, buff=0.5)
        k_label = Text("Keys (K)", font_size=16, color=K_COLOR)
        k_group = VGroup(k_vecs, k_label).arrange(UP, buff=0.2)
        # Fix Issue 29: Move k_group to C4
        self.place_at_grid(k_group, "C4")

        self.play(FadeIn(q_group), FadeIn(k_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Lecture line color change
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )

        # Connection lines
        connections = VGroup()
        scores = VGroup()
        
        for i, target_k in enumerate(k_vecs):
            line = Line(q_vec.get_right(), target_k.get_left(), stroke_width=2, color=GREY)
            connections.add(line)
            
            score_val = [12, 50, 8][i]
            score_box = Rectangle(height=0.3, width=0.5, color=SCORE_BOX_COLOR)
            score_text = Text(str(score_val), font_size=14)
            score_item = VGroup(score_box, score_text)
            score_item.move_to(line.get_center())
            scores.add(score_item)

        self.play(Create(connections))
        self.play(FadeIn(scores))
        self.wait(0.5)

        # Highlight the match (middle one)
        high_score_line = connections[1]
        high_score_box = scores[1]
        
        self.play(
            high_score_line.animate.set_stroke(color=HIGHLIGHT_COLOR, width=5),
            high_score_box[0].animate.set_color(HIGHLIGHT_COLOR),
            high_score_box[1].animate.set_color(HIGHLIGHT_COLOR).scale(1.2)
        )
        self.play(Flash(high_score_box, color=HIGHLIGHT_COLOR, flash_radius=0.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Lecture line color change
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(ORANGE)
        )

        # Formula: Score / sqrt(dk)
        formula = MathTex(r"\text{Score} = \frac{Q \cdot K^T}{\sqrt{d_k}}", font_size=24, color=WHITE)
        # Fix Issue 30: Move formula to E2-E4
        self.place_in_area(formula, "E2", "E4")
        
        # Numeric scaling example
        val_50 = Text("50", font_size=24, color=HIGHLIGHT_COLOR)
        divide = Text("/", font_size=24)
        val_sqrt = Text("10", font_size=24)
        equals = Text("=", font_size=24)
        val_5 = Text("5", font_size=24, color=HEATMAP_COLORS[0])
        
        scale_example = VGroup(val_50, divide, val_sqrt, equals, val_5).arrange(RIGHT, buff=0.2)
        # Fix Issue 30: Move scale_example to F2-F4
        self.place_in_area(scale_example, "F2", "F4")

        self.play(Write(formula))
        self.play(FadeIn(scale_example))
        self.wait(1)

        # Heatmap grid
        heatmap = VGroup()
        for i in range(3):
            for j in range(3):
                color_idx = (i + j) % 3
                rect = Rectangle(height=0.4, width=0.4, fill_color=HEATMAP_COLORS[color_idx], fill_opacity=0.8, stroke_width=1)
                heatmap.add(rect)
        heatmap.arrange_in_grid(rows=3, cols=3, buff=0.05)
        # Fix Issue 31: Move heatmap to C5-E6
        self.place_in_area(heatmap, "C5", "E6")
        
        heatmap_label = Text("Attention Scores (Heatmap)", font_size=16, color=ORANGE)
        # Fix Issue 31: Move heatmap_label to B5-B6
        self.place_in_area(heatmap_label, "B5", "B6")
        
        self.play(FadeIn(heatmap_label), LaggedStartMap(FadeIn, heatmap, shift=UP*0.2))
        self.wait(2)
