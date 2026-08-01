from manim import *
import math

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
        # Helper function for binomial probability
        def get_binomial_probs(n, p):
            # Using math.comb for reliability
            return [math.comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in range(n + 1)]

        n = 10
        probs_05 = get_binomial_probs(n, 0.5)
        probs_07 = get_binomial_probs(n, 0.7)

        # 1. Setup Layout
        self.setup_layout(
            "Visualizing the Distribution",
            [
                "A histogram shows probabilities for all possible outcomes.",
                "Changing 'p' shifts the distribution's shape and peak.",
                "The tallest bar shows the most likely outcome."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#ADD8E6")
        
        # Create BarChart for n=10, p=0.5 in Blue (#ADD8E6)
        # Using a height of 3.5 and width 5 fits well in the B2-F6 area
        chart = BarChart(
            values=probs_05,
            bar_names=[str(i) for i in range(n+1)],
            y_range=[0, 0.4, 0.1],
            x_length=5,
            y_length=3.5,
            bar_colors=["#ADD8E6"] * (n+1),
            axis_config={"color": "#FFFFFF", "font_size": 24}
        )
        
        # Position in a safe area (B2 to F6) to avoid overlap with lecture text (L002)
        # This placement resolves alignment/overlap issues (Issue #25, #26)
        self.place_in_area(chart, 'B2', 'F6', scale_factor=0.8)
        
        self.play(Create(chart))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#90EE90")
        
        # Smoothly morph BarChart to n=10, p=0.7 in Green (#90EE90)
        # change_bar_values is an animation-friendly method for BarChart
        self.play(
            chart.animate.change_bar_values(probs_07),
            chart.bars.animate.set_fill("#90EE90"),
            run_time=2.0
        )
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        
        # Highlight bar at k=7 with Flash and color #FF4500
        # k=7 is the 8th bar (index 7)
        bar_to_highlight = chart.bars[7]
        
        self.play(
            Flash(bar_to_highlight, color="#FF4500", flash_radius=0.5),
            bar_to_highlight.animate.set_fill("#FF4500"),
            run_time=1.5
        )
        self.wait(1.5)
