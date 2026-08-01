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
        # Data from shared state/coordination
        title_text = "The Law of Large Numbers (Sample Size n)"
        lecture_lines = [
            "Small samples produce wide, short distributions.",
            "This spread reflects higher uncertainty in our estimate.",
            "Increasing n makes the distribution taller and thinner.",
            "This results in much higher precision around the mean.",
            "We quantify this spread using the Standard Error formula.",
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_WIDE = "#FFDAB9"  # n=2
        COLOR_NARROW = "#87CEFA" # n=50
        COLOR_PRECISION = "#00FF00"
        COLOR_FORMULA = "#FFFF00"
        COLOR_HIGHLIGHT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Plot for n=2 (Small sample size)
        # Issue 49: Adjust positioning to Column 1 to avoid overlap with n=50 axes
        axes_left = Axes(
            x_range=[-3, 3, 1], y_range=[0, 1, 0.5],
            x_length=2.5, y_length=1.5,
            axis_config={"include_tip": False, "color": GREY}
        )
        self.place_in_area(axes_left, "C1", "D2", scale_factor=1.0)
        
        curve_wide = axes_left.plot(
            lambda x: np.exp(-0.5 * (x / 1.2)**2) / (1.2 * np.sqrt(2 * np.pi)),
            color=COLOR_WIDE
        )
        
        # Issue 49: Move label to Column 1
        label_n2 = Text("n=2", color=COLOR_WIDE, font_size=24)
        self.place_at_grid(label_n2, "B1", scale_factor=1.0)
        
        self.play(self.lecture[0].animate.set_color(COLOR_WIDE))
        self.play(Create(axes_left), Create(curve_wide), Write(label_n2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight wide spread (Uncertainty)
        # Issue 49: Move label to Column 1
        arrow_wide = DoubleArrow(
            axes_left.c2p(-1.5, 0.1), axes_left.c2p(1.5, 0.1),
            color=COLOR_HIGHLIGHT, buff=0, tip_length=0.1
        )
        text_uncertainty = Text("Uncertain", color=COLOR_HIGHLIGHT, font_size=16)
        self.place_at_grid(text_uncertainty, "E1", scale_factor=1.0)
        
        self.play(self.lecture[1].animate.set_color(COLOR_HIGHLIGHT))
        self.play(GrowFromCenter(arrow_wide), FadeIn(text_uncertainty))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Plot for n=50 (Large sample size)
        axes_right = Axes(
            x_range=[-3, 3, 1], y_range=[0, 1, 0.5],
            x_length=2.5, y_length=1.5,
            axis_config={"include_tip": False, "color": GREY}
        )
        self.place_in_area(axes_right, "C4", "D5", scale_factor=1.0)
        
        curve_narrow = axes_right.plot(
            lambda x: np.exp(-0.5 * (x / 0.4)**2) / (0.4 * np.sqrt(2 * np.pi)),
            color=COLOR_NARROW
        )
        
        # Issue 50: Move label to B5 to avoid obstruction by the tall curve peak
        label_n50 = Text("n=50", color=COLOR_NARROW, font_size=24)
        self.place_at_grid(label_n50, "B5", scale_factor=1.0)

        self.play(self.lecture[2].animate.set_color(COLOR_NARROW))
        self.play(Create(axes_right), Create(curve_narrow), Write(label_n50))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight narrow spread (Precision)
        arrow_narrow = DoubleArrow(
            axes_right.c2p(-0.5, 0.1), axes_right.c2p(0.5, 0.1),
            color=COLOR_PRECISION, buff=0, tip_length=0.08
        )
        text_precision = Text("Precise", color=COLOR_PRECISION, font_size=16)
        self.place_at_grid(text_precision, "E4", scale_factor=1.0)
        
        self.play(self.lecture[3].animate.set_color(COLOR_PRECISION))
        self.play(GrowFromCenter(arrow_narrow), FadeIn(text_precision))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Standard Error Formula
        # Issue 51: Reduce scale_factor to 0.9 to prevent crowding
        formula = Text("SE = σ / √n", color=COLOR_FORMULA, font_size=32)
        self.place_in_area(formula, "F2", "F5", scale_factor=0.9)
        
        self.play(self.lecture[4].animate.set_color(COLOR_FORMULA))
        self.play(Write(formula))
        self.wait(2)
