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
        self.setup_layout(
            "The Experiment: Repeated Sampling",
            [
                "Let's take many different samples from our messy population.",
                "Each sample contains a specific number of random individuals.",
                "For every group, we calculate its unique sample mean.",
                "We record these means on a new frequency graph.",
                "Initially, the resulting distribution might still look scattered."
            ]
        )
        
        # Define colors for lecture lines and elements
        c1 = "#FFFFFF" # White (Uniform Plot)
        c2 = "#FFFF00" # Yellow (Bracket)
        c3 = "#87CEEB" # Sky Blue (Single Mean Dot)
        c4 = "#98FB98" # Pale Green (Bottom Axis)
        c5 = "#FFDAB9" # Peach Puff (Multiple Dots)

        # === Animation for Lecture Line 1 ===
        # A flat uniform distribution plot appears in white (#FFFFFF) at the top.
        self.play(self.lecture[0].animate.set_color(c1))
        
        top_axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 2, 1],
            x_length=5,
            y_length=2,
            axis_config={"include_tip": False, "color": c1}
        ).set_z_index(0)
        # Reduced scale factor to 0.7 to avoid crowding as per Issue 26
        self.place_in_area(top_axes, "A1", "C6", scale_factor=0.7)
        
        uniform_line = Line(
            top_axes.c2p(1, 1), top_axes.c2p(9, 1), color=c1, stroke_width=4
        ).set_z_index(1)
        
        # Represent "messy population" with random dots
        np.random.seed(42) 
        population_dots = VGroup(*[
            Dot(top_axes.c2p(np.random.uniform(1.2, 8.8), np.random.uniform(0.1, 0.9)), 
                radius=0.03, color=c1, fill_opacity=0.6)
            for _ in range(80)
        ]).set_z_index(1)
        
        self.play(Create(top_axes), Create(uniform_line))
        self.play(FadeIn(population_dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A yellow (#FFFF00) bracket highlights a small group of data points in the plot.
        self.play(self.lecture[1].animate.set_color(c2))
        
        bracket1 = BraceBetweenPoints(top_axes.c2p(2, -0.1), top_axes.c2p(4, -0.1), color=c2)
        sample_label = Text("Sample 1", font_size=16, color=c2).next_to(bracket1, DOWN, buff=0.1)
        
        # Identify dots within the range [2, 4]
        highlighted_dots1 = VGroup(*[
            dot for dot in population_dots if 2 <= top_axes.p2c(dot.get_center())[0] <= 4
        ])
        
        self.play(Create(bracket1), Write(sample_label))
        self.play(highlighted_dots1.animate.set_color(c2).set_scale(1.5))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # For every group, we calculate its unique sample mean.
        self.play(self.lecture[2].animate.set_color(c3))
        
        mean_val1 = 3.0
        mean_dot1 = Dot(top_axes.c2p(mean_val1, 1.2), color=c1, radius=0.08).set_z_index(2)
        mean_label = MathTex(r"\bar{x}_1", font_size=20, color=c1).next_to(mean_dot1, UP, buff=0.1)
        
        self.play(FadeIn(mean_dot1), Write(mean_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # We record these means on a new frequency graph.
        self.play(self.lecture[3].animate.set_color(c4))
        
        bottom_axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 5, 1],
            x_length=5,
            y_length=2,
            axis_config={"include_tip": False, "color": c4}
        ).set_z_index(0)
        # Reduced scale factor to 0.7 to avoid crowding as per Issue 27
        self.place_in_area(bottom_axes, "D1", "F6", scale_factor=0.7)
        
        dest_pos1 = bottom_axes.c2p(mean_val1, 0.4)
        
        self.play(Create(bottom_axes))
        self.play(
            mean_dot1.animate.move_to(dest_pos1),
            FadeOut(mean_label),
            FadeOut(bracket1),
            FadeOut(sample_label),
            highlighted_dots1.animate.set_color(c1).set_scale(1/1.5)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Initially, the resulting distribution might still look scattered.
        self.play(self.lecture[4].animate.set_color(c5))
        
        # Second sample
        bracket2 = BraceBetweenPoints(top_axes.c2p(6, -0.1), top_axes.c2p(8, -0.1), color=c2)
        sample_label2 = Text("Sample 2", font_size=16, color=c2).next_to(bracket2, DOWN, buff=0.1)
        highlighted_dots2 = VGroup(*[
            dot for dot in population_dots if 6 <= top_axes.p2c(dot.get_center())[0] <= 8
        ])
        
        mean_val2 = 7.0
        mean_dot2 = Dot(top_axes.c2p(mean_val2, 1.2), color=c1, radius=0.08).set_z_index(2)
        
        self.play(Create(bracket2), Write(sample_label2), highlighted_dots2.animate.set_color(c2).set_scale(1.5))
        self.play(FadeIn(mean_dot2))
        
        dest_pos2 = bottom_axes.c2p(mean_val2, 0.4)
        self.play(
            mean_dot2.animate.move_to(dest_pos2),
            FadeOut(bracket2),
            FadeOut(sample_label2),
            highlighted_dots2.animate.set_color(c1).set_scale(1/1.5)
        )
        
        # Rapid drop of multiple means
        num_additional = 15
        random_means = np.random.uniform(1.5, 8.5, num_additional)
        
        bin_counts = {}
        for m in [3.0, 7.0]:
            b = np.round(m, 0)
            bin_counts[b] = bin_counts.get(b, 0) + 1
            
        stack_anims = []
        for rm in random_means:
            b = np.round(rm, 0)
            count = bin_counts.get(b, 0)
            bin_counts[b] = count + 1
            
            start_dot = Dot(top_axes.c2p(rm, 0.5), color=c1, radius=0.05).set_z_index(2)
            end_pos = bottom_axes.c2p(b, 0.4 + count * 0.25)
            
            stack_anims.append(
                Succession(
                    FadeIn(start_dot, run_time=0.1),
                    start_dot.animate(run_time=0.4).move_to(end_pos)
                )
            )
            
        self.play(AnimationGroup(*stack_anims, lag_ratio=0.15))
        self.wait(2)
