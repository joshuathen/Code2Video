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
        # Setup the basic layout
        title = "The Core Rules of CLT"
        lines = [
            "The curve's center matches the true population mean.",
            "Larger samples create a narrower, more precise curve.",
            "Usually, thirty or more samples ensure this symmetry."
        ]
        self.setup_layout(title, lines)

        # Helper function for Normal Distribution curve
        def normal_pdf(x, mu, sigma):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)

        # Setup Axes in the animation area (B2 to E5)
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 1.2, 0.5],
            x_length=4,
            y_length=3,
            axis_config={"include_tip": False, "color": WHITE},
            tips=False
        )
        self.place_in_area(axes, "B2", "E5")

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(WHITE), run_time=0.5)

        # Baseline curve (moderate sigma)
        curve_base = axes.plot(lambda x: normal_pdf(x, 0, 0.8), color=WHITE, x_range=[-4, 4])
        
        # Dashed line and Mean label
        peak_x = 0
        peak_y = normal_pdf(peak_x, 0, 0.8)
        dashed_line = DashedLine(
            start=axes.c2p(peak_x, peak_y),
            end=axes.c2p(peak_x, 0),
            color=WHITE
        )
        mu_label = Text("μ", color=WHITE)
        # Fix Issue 48: Align mu_label between F3 and F4
        self.place_in_area(mu_label, "F3", "F4", scale_factor=0.8)

        self.play(Create(axes), Create(curve_base))
        self.play(Create(dashed_line), Write(mu_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(self.lecture[1].animate.set_color("#32CD32"), run_time=0.5)

        # Define Wide curve (n=5 -> larger sigma) and Narrow curve (n=50 -> smaller sigma)
        curve_wide = axes.plot(lambda x: normal_pdf(x, 0, 1.0), color="#FF6347", x_range=[-4, 4])
        curve_narrow = axes.plot(lambda x: normal_pdf(x, 0, 0.35), color="#32CD32", x_range=[-4, 4])
        
        label_n5 = Text("n = 5", font_size=20, color="#FF6347")
        label_n50 = Text("n = 50", font_size=20, color="#32CD32")
        
        # Position labels near the curves
        # Fix Issue 47: Move n=5 label closer to the curve peak vertically
        self.place_at_grid(label_n5, "C5", scale_factor=0.8)
        self.place_at_grid(label_n50, "B3", scale_factor=0.8)

        # Transition: Remove base curve, add the two comparison curves
        self.play(
            FadeOut(curve_base),
            Create(curve_wide),
            Create(curve_narrow),
            Write(label_n5),
            Write(label_n50)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(self.lecture[2].animate.set_color("#FFD700"), run_time=0.5)

        # Magic Threshold Text
        threshold_text = Text("Magic Threshold: n ≥ 30", color="#FFD700", weight=BOLD)
        # Fix Issue 49: Scale down slightly to avoid cramping
        self.place_in_area(threshold_text, "A2", "A5", scale_factor=0.75)
        
        # Add highlight or box for the threshold text
        surround_rect = SurroundingRectangle(threshold_text, color="#FFD700", buff=0.2)

        self.play(Write(threshold_text))
        self.play(Create(surround_rect))
        self.wait(2)
