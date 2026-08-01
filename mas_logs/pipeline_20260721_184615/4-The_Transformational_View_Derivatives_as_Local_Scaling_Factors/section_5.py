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
        # Initializing layout with fetched title and lecture lines
        title = "The Chain Rule: Compounding Stretches"
        lines = [
            "We can view composition using three parallel number lines.",
            "The first transformation scales the input space by three.",
            "The second transformation scales that result by two.",
            "Multiplying these factors gives a total scaling of six."
        ]
        self.setup_layout(title, lines)

        # Colors from storyboard
        color_g = "#ADFF2F"
        color_f = "#FF6347"
        color_total = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Show three parallel lines X, U, and Y representing domains/ranges
        self.lecture[0].set_color(WHITE)
        
        line_x = Line(self.grid["B2"], self.grid["B6"], color=WHITE)
        label_x = MathTex("X", color=WHITE, font_size=32)
        self.place_at_grid(label_x, "B1", scale_factor=0.6) # Resolved Issue 32
        
        line_u = Line(self.grid["D2"], self.grid["D6"], color=WHITE)
        label_u = MathTex("U", color=WHITE, font_size=32)
        self.place_at_grid(label_u, "D1", scale_factor=0.6) # Resolved Issue 32
        
        line_y = Line(self.grid["F2"], self.grid["F6"], color=WHITE)
        label_y = MathTex("Y", color=WHITE, font_size=32)
        self.place_at_grid(label_y, "F1", scale_factor=0.6) # Resolved Issue 32
        
        self.play(Create(line_x), FadeIn(label_x))
        self.play(Create(line_u), FadeIn(label_u))
        self.play(Create(line_y), FadeIn(label_y))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate mapping X -> U with 'g'(x) = 3'
        self.play(self.lecture[1].animate.set_color(color_g))
        
        # Segment on X: initial length 0.4 units
        # Center of grid B2 is (1.5, 1.2). We'll offset it.
        start_x = self.grid["B2"] + 0.5 * RIGHT
        end_x = start_x + 0.4 * RIGHT
        seg_x = Line(start_x, end_x, color=color_g, stroke_width=8)
        dot_x1 = Dot(start_x, color=color_g, radius=0.08)
        dot_x2 = Dot(end_x, color=color_g, radius=0.08)
        
        # Segment on U: stretched length 1.2 units (3x scaling)
        start_u = self.grid["D2"] + 0.5 * RIGHT
        end_u = start_u + 1.2 * RIGHT
        seg_u = Line(start_u, end_u, color=color_g, stroke_width=8)
        dot_u1 = Dot(start_u, color=color_g, radius=0.08)
        dot_u2 = Dot(end_u, color=color_g, radius=0.08)
        
        # Visualizing the mapping
        arrow_g1 = CurvedArrow(start_x, start_u, angle=-TAU/12, color=color_g)
        arrow_g2 = CurvedArrow(end_x, end_u, angle=-TAU/12, color=color_g)
        
        label_g = MathTex(r"g'(x) = 3", color=color_g, font_size=34)
        self.place_at_grid(label_g, "C2", scale_factor=0.7) # Resolved Issue 31

        self.play(Create(seg_x), FadeIn(dot_x1), FadeIn(dot_x2))
        self.play(Create(arrow_g1), Create(arrow_g2))
        self.play(Create(seg_u), FadeIn(dot_u1), FadeIn(dot_u2), Write(label_g))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate mapping U -> Y with 'f'(u) = 2'
        self.play(self.lecture[2].animate.set_color(color_f))
        
        # Segment on Y: stretched length 2.4 units (2x scaling from U's 1.2)
        start_y = self.grid["F2"] + 0.5 * RIGHT
        end_y = start_y + 2.4 * RIGHT
        seg_y = Line(start_y, end_y, color=color_f, stroke_width=8)
        dot_y1 = Dot(start_y, color=color_f, radius=0.08)
        dot_y2 = Dot(end_y, color=color_f, radius=0.08)
        
        arrow_f1 = CurvedArrow(start_u, start_y, angle=-TAU/12, color=color_f)
        arrow_f2 = CurvedArrow(end_u, end_y, angle=-TAU/12, color=color_f)
        
        label_f = MathTex(r"f'(u) = 2", color=color_f, font_size=34)
        self.place_at_grid(label_f, "E2", scale_factor=0.7) # Resolved Issue 31

        self.play(Create(arrow_f1), Create(arrow_f2))
        self.play(Create(seg_y), FadeIn(dot_y1), FadeIn(dot_y2), Write(label_f))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Summarize the total scaling factor (3 * 2 = 6)
        self.play(self.lecture[3].animate.set_color(color_total))
        
        # Highlighting the compound mapping from X directly to Y
        arrow_total = Arrow(self.grid["B6"] + 0.5 * RIGHT, self.grid["F6"] + 0.5 * RIGHT, color=color_total, buff=0.1)
        label_total = MathTex(r"Total\ Scaling = 3 \times 2 = 6", color=color_total, font_size=32)
        self.place_in_area(label_total, "C3", "E5", scale_factor=0.7) # Resolved Issue 30

        self.play(Create(arrow_total))
        self.play(Write(label_total))
        self.wait(1)

        # Display formula dy/dx = (dy/du) * (du/dx)
        formula = MathTex(
            r"\frac{dy}{dx} =", r"\frac{dy}{du}", r"\cdot", r"\frac{du}{dx}",
            color=color_total, font_size=36
        )
        self.place_in_area(formula, "E4", "F6", scale_factor=0.8)
        
        self.play(Write(formula))
        self.play(Indicate(formula[2], color=WHITE, scale_factor=1.5)) # Flash multiplication sign
        self.wait(2)
