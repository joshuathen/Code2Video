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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "Defining the PDF: It's Not a Point, It's a Shape"
        lines = [
            "This smooth shape is the Probability Density Function.",
            "The vertical axis represents density, not direct probability.",
            "High peaks show where data points crowd together.",
            "Flat valleys represent less likely measurement outcomes.",
            "Together, they form a map of likelihood."
        ]
        self.setup_layout(title, lines)

        # Colors
        PDF_COLOR = "#ADD8E6"  # Light Blue
        AXIS_LABEL_COLOR = "#D3D3D3"  # Light Gray
        FORMULA_COLOR = "#FFFFFF"
        DOT_COLOR = "#FFD700"  # Gold
        HIGHLIGHT_COLOR = "#FFFF00"

        # Define PDF function (Normal Distribution)
        def pdf_func(x):
            return 2 * np.exp(-0.5 * (x)**2)

        # === Animation for Lecture Line 1 ===
        # "This smooth shape is the Probability Density Function."
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Create Axes in area B3-F6 to avoid left edge cutoff (Issue 27)
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 2.5, 0.5],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": AXIS_LABEL_COLOR}
        )
        self.place_in_area(axes, "B3", "F6", scale_factor=0.8)
        
        # Labels for axes
        y_label = Text("Density", font_size=20, color=AXIS_LABEL_COLOR).scale(0.8)
        x_label = Text("Value", font_size=20, color=AXIS_LABEL_COLOR).scale(0.8)
        
        # Proximity positioning (L002)
        y_label.next_to(axes.y_axis.get_top(), LEFT, buff=0.1)
        x_label.next_to(axes.x_axis.get_right(), DOWN, buff=0.1)

        curve = axes.plot(pdf_func, color=PDF_COLOR, x_range=[-3, 3])
        
        # Formula placement at B5 to avoid overlap (Issue 28)
        formula = MathTex("f(x)", color=FORMULA_COLOR)
        self.place_at_grid(formula, "B5", scale_factor=0.7)

        self.play(Create(axes), Write(y_label), Write(x_label))
        self.play(Create(curve), Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The vertical axis represents density, not direct probability."
        self.lecture[0].set_color("#FFFFFF")
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        self.play(Indicate(y_label, color=HIGHLIGHT_COLOR))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "High peaks show where data points crowd together."
        self.lecture[1].set_color("#FFFFFF")
        self.lecture[2].set_color(HIGHLIGHT_COLOR)

        # Generate dots concentrated at the peak
        # Using Dot with set_fill to be safe (L031)
        peak_dots = VGroup(*[
            Dot(axes.c2p(np.random.normal(0, 0.5), np.random.uniform(0, 0.2)), radius=0.03, color=DOT_COLOR)
            .set_fill(opacity=0.8)
            for _ in range(50)
        ])
        
        peak_indicator = Dot(axes.c2p(0, pdf_func(0)), radius=0.08, color=HIGHLIGHT_COLOR)
        
        self.play(FadeIn(peak_dots, shift=UP * 0.2))
        self.play(Indicate(peak_indicator, scale_factor=2.0))
        self.remove(peak_indicator)
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # "Flat valleys represent less likely measurement outcomes."
        self.lecture[2].set_color("#FFFFFF")
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Generate dots in the tails
        tail_dots = VGroup(*[
            Dot(axes.c2p(x, np.random.uniform(0, 0.1)), radius=0.03, color=DOT_COLOR)
            .set_fill(opacity=0.8)
            for x in [np.random.uniform(-2.8, -1.8) for _ in range(10)] + [np.random.uniform(1.8, 2.8) for _ in range(10)]
        ])
        
        self.play(FadeIn(tail_dots, shift=UP * 0.1))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # "Together, they form a map of likelihood."
        self.lecture[3].set_color("#FFFFFF")
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        all_elements = VGroup(curve, peak_dots, tail_dots, formula)
        self.play(Indicate(all_elements))
        self.wait(3)
