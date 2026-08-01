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
        # Setup layout
        lecture_lines = [
            "First, the PDF must never drop below zero.",
            "Second, the total area must always equal exactly one.",
            "This ensures the total probability is exactly 100 percent."
        ]
        self.setup_layout("The Golden Rules of PDFs", lecture_lines)

        # Create Axes and Graph
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.2, 0.5],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE},
        )
        self.place_in_area(axes, "B1", "E6", scale_factor=1.0)
        
        # Define a PDF-like curve (Gaussian)
        # f(x) = exp(-x^2/2) / sqrt(2*pi)
        def pdf_func(x):
            return 0.9 * np.exp(-0.5 * (x)**2)

        curve = axes.plot(pdf_func, x_range=[-3, 3], color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Line 1: "First, the PDF must never drop below zero."
        # Highlight: f(x) >= 0
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Create green label f(x) >= 0 (Issue 50: Moved to B6 and scaled)
        non_neg_label = Text("f(x) \u2265 0", color="#00FF00", font_size=32)
        self.place_at_grid(non_neg_label, "B6", scale_factor=0.8)
        
        # Animation: Curve and Axes appear, then highlight curve green
        self.play(Create(axes), Create(curve), run_time=1.5)
        self.play(curve.animate.set_stroke(color="#00FF00", width=4))
        self.play(Write(non_neg_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: "Second, the total area must always equal exactly one."
        # Highlight: Shade area under curve in light blue
        self.play(self.lecture[1].animate.set_color("#ADD8E6"))
        
        # Shade entire area under curve
        area = axes.get_area(curve, x_range=[-3, 3], color="#ADD8E6", opacity=0.4)
        
        self.play(FadeIn(area, shift=UP), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: "This ensures the total probability is exactly 100 percent."
        # Highlight: Golden counter 'Total Area = 1'
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        # Golden counter display (Issue 38: Asset integration; Issue 51: Balanced area placement)
        counter_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/counter.svg", color="#FFD700")
        counter_text = Text("Total Area = 1", color="#FFD700", font_size=36)
        
        # Scale icon to match text height then group
        counter_icon.scale_to_fit_height(counter_text.height)
        counter_group = VGroup(counter_icon, counter_text).arrange(RIGHT, buff=0.3)
        
        self.place_in_area(counter_group, "F2", "F5", scale_factor=0.8)
        
        self.play(
            FadeIn(counter_group, scale=1.2),
            counter_group.animate.set_stroke(color="#FFD700", width=1),
            run_time=1.5
        )
        self.wait(2)
