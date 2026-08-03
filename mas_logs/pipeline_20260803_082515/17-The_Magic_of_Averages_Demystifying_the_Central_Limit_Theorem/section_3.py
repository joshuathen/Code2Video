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
        # Data and Setup
        np.random.seed(42)
        title_text = "The Experiment: The Sampling Process"
        lecture_lines = [
            "First, we take a random sample of size n.",
            "Next, we calculate the average for that sample.",
            "We repeat this process hundreds or thousands of times.",
            "Each sample mean is plotted on a new graph.",
            "This builds the sampling distribution of the mean."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Pre-create main visual elements for Issue fixes
        graph_axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1, 0.2],
            x_length=6,
            y_length=4,
            axis_config={"include_tip": False, "font_size": 20, "color": WHITE}
        )
        # Issue 29 Fix: Position graph_axes at B3-E6
        self.place_in_area(graph_axes, 'B3', 'E6', scale_factor=0.8)
        
        # Issue 27 Fix: Position prob_label at A2
        prob_label = Text("Prob", font_size=24, color=WHITE)
        self.place_at_grid(prob_label, 'A2', scale_factor=0.7)

        # Issue 28 Fix: Position normal_label at F3-F6
        normal_label = Text("Sampling Dist. of Sample Means (Normal)", font_size=24, color=GREEN)
        self.place_in_area(normal_label, 'F3', 'F6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        # Highlight: First, we take a random sample of size n.
        self.lecture[0].set_color(WHITE)
        
        # Source "dots" representing sample selection
        source_dots = VGroup(*[Dot(radius=0.06, color=BLUE_A) for _ in range(30)])
        for dot in source_dots:
            dot.move_to(self.grid["A5"] + np.array([np.random.uniform(-1.2, 1.2), np.random.uniform(-0.4, 0.4), 0]))
        
        self.play(Create(graph_axes), Write(prob_label))
        self.play(LaggedStartMap(FadeIn, source_dots, lag_ratio=0.02))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight: Next, we calculate the average for that sample.
        self.lecture[1].set_color(YELLOW)
        
        avg_calc_text = MathTex(r"\text{Average } \bar{x} = 5.2", color=YELLOW)
        self.place_at_grid(avg_calc_text, "B1", scale_factor=0.8)
        
        self.play(Write(avg_calc_text))
        self.play(Flash(avg_calc_text, color=YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight: We repeat this process hundreds or thousands of times.
        self.lecture[2].set_color(PINK)
        
        # Transition: Current sample dots collapse into the average, then into a graph dot
        mean_dot_initial = Dot(radius=0.06, color=WHITE)
        mean_dot_initial.move_to(graph_axes.c2p(5.2, 0.05))
        
        self.play(
            source_dots.animate.move_to(avg_calc_text.get_center()).set_opacity(0),
            ReplacementTransform(avg_calc_text, mean_dot_initial),
            run_time=1.5
        )
        self.remove(source_dots)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Highlight: Each sample mean is plotted on a new graph.
        self.lecture[3].set_color(WHITE)
        
        all_mean_dots = VGroup(mean_dot_initial)
        num_iterations = 60
        bin_counts = {}

        def get_stacked_pos(val):
            # Simple binning for visual stacking of dots on the x-axis
            b = round(val, 1)
            count = bin_counts.get(b, 0)
            bin_counts[b] = count + 1
            return graph_axes.c2p(val, count * 0.04 + 0.05)

        # Re-set first dot position with stacking logic
        mean_dot_initial.move_to(get_stacked_pos(5.2))

        # Show rapid sampling repetition
        for i in range(num_iterations):
            val = np.random.normal(5, 0.8)
            val = np.clip(val, 1, 9)
            new_dot = Dot(radius=0.04, color=WHITE).move_to(get_stacked_pos(val))
            all_mean_dots.add(new_dot)
            self.add(new_dot)
            # Speed up the wait time as more dots appear
            wait_time = max(0.01, 0.1 * (0.85 ** i))
            self.wait(wait_time)
        
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight: This builds the sampling distribution of the mean.
        self.lecture[4].set_color(GREEN)
        
        # Final bell curve showing the sampling distribution
        dist_curve = graph_axes.plot(
            lambda x: 0.8 * np.exp(-0.5 * ((x - 5) / 0.8)**2),
            color=GREEN,
            x_range=[2, 8]
        )
        
        self.play(
            Create(dist_curve),
            Write(normal_label),
            all_mean_dots.animate.set_color(GREEN).set_opacity(0.4),
            run_time=2
        )
        self.wait(3)
