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

class Section2Scene(TeachingScene):
    def construct(self):
        # Initialization
        title_text = "Prerequisite: The Histogram and the Mean"
        lecture_lines = [
            "A histogram visualizes the frequency of data points.",
            "The mean represents the data's central balancing point.",
            "We focus on averages of groups, or sample means."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        HIST_COLOR = "#00BFFF"
        MEAN_COLOR = WHITE
        SAMPLE_COLOR = "#FFFF00" # Yellow for sample mean

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIST_COLOR))
        
        # Create Axes for Histogram
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 6, 2],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False, "color": GREY},
            tips=False
        )
        # Issue 44: Scale factor 0.85 to improve margin
        self.place_in_area(axes, "B2", "F6", scale_factor=0.85)
        
        # Issue 36: Integration of asset
        creature_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/creature.svg")
        creature_icon.set_color(HIST_COLOR)
        self.place_at_grid(creature_icon, "B1", scale_factor=0.5)

        # Create Bimodal Histogram Bars (Creature Heights)
        bar_data = [
            (1, 1), (2, 4), (3, 1), # First peak
            (7, 1), (8, 4), (9, 1)  # Second peak
        ]
        
        bars = VGroup()
        for x_val, height in bar_data:
            bar_w = (axes.x_axis.get_unit_size() * 0.8)
            bar_h = (axes.y_axis.get_unit_size() * height)
            rect = Rectangle(
                width=bar_w,
                height=bar_h,
                fill_color=HIST_COLOR,
                fill_opacity=0.7,
                stroke_color=WHITE,
                stroke_width=1
            )
            rect.move_to(axes.c2p(x_val, height / 2))
            bars.add(rect)

        self.play(Create(axes), Create(bars), FadeIn(creature_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(MEAN_COLOR))
        
        # Population mean line
        mean_x = 5
        mean_line = Line(
            axes.c2p(mean_x, 0),
            axes.c2p(mean_x, 5),
            color=MEAN_COLOR,
            stroke_width=6
        )
        
        # Label: 'Population Mean (μ)'
        mean_label = Text("Population Mean (μ)", font_size=24, color=MEAN_COLOR)
        # Issue 43: Use place_in_area for centering
        self.place_in_area(mean_label, 'A3', 'A6', scale_factor=0.8)
        
        self.play(Create(mean_line), Write(mean_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(SAMPLE_COLOR))
        
        # Sample Mean dot
        sample_mean_x = 5.8 
        sample_dot = Dot(axes.c2p(sample_mean_x, 3), color=SAMPLE_COLOR, radius=0.12)
        sample_dot_label = Text("Sample Mean", font_size=20, color=SAMPLE_COLOR)
        # Issue 42: Fixed positioning to avoid overlap
        self.place_at_grid(sample_dot_label, 'C5', scale_factor=0.7)
        
        # Transition: Bars fade, focus on dot
        self.play(
            bars.animate.set_opacity(0.2),
            FadeIn(sample_dot),
            Write(sample_dot_label)
        )
        
        # Animate sample mean point moving to show variation
        self.play(
            sample_dot.animate.move_to(axes.c2p(4.2, 3)),
            run_time=2,
            rate_func=wiggle
        )
        
        self.wait(2)
