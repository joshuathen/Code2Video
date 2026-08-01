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
        # Setup Layout
        lines = [
            "Let's sample heights from these wild alien monsters.",
            "Our machine takes fifty random monsters per scoop.",
            "It calculates the mean height for every single scoop.",
            "Each mean becomes a dot on our results graph.",
            "Watch as the dots begin to pile up."
        ]
        self.setup_layout("The Experiment: Sampling Distribution", lines)
        
        # Define Colors from Prompt
        color_pop = "#9575CD"    # Purple for Population
        color_select = "#FFFFFF" # White for Selection
        color_mean = "#FFD54F"   # Yellow for Mean/Results
        
        # Lecture line color mapping
        color1 = color_pop
        color2 = color_select
        color3 = color_mean
        color4 = color_mean
        color5 = color_mean

        # === Population Axis (Shark Fin) ===
        # Fix Issue 40: scale 0.7 in A1-B6
        axes_pop = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 0.5, 0.1],
            axis_config={"include_tip": False, "font_size": 18, "label_constructor": Text},
            x_length=5,
            y_length=2
        ).add_coordinates()
        pop_label = Text("Population Distribution", font_size=18).next_to(axes_pop, UP, buff=0.1)
        pop_group = VGroup(axes_pop, pop_label)
        self.place_in_area(pop_group, "A1", "B6", scale_factor=0.7)

        # Skewed Distribution (Shark Fin: Gamma-like)
        def shark_fin(x):
            return 0.4 * (x**1.5) * np.exp(-x)
        
        curve = axes_pop.plot(shark_fin, x_range=[0, 8], color=color_pop)
        area = axes_pop.get_area(curve, x_range=[0, 8], color=color_pop, opacity=0.3)

        # === Sampling Axis ===
        # Fix Issue 39: Move to E1-F6, scale 0.8
        axes_samp = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            axis_config={"include_tip": False, "font_size": 18, "label_constructor": Text},
            x_length=5,
            y_length=2
        ).add_coordinates()
        samp_label = Text("Sampling Distribution of Means", font_size=18).next_to(axes_samp, UP, buff=0.1)
        samp_group = VGroup(axes_samp, samp_label)
        self.place_in_area(samp_group, "E1", "F6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color1)
        self.play(Create(axes_pop), Write(pop_label))
        self.play(Create(curve), FadeIn(area))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color2)
        
        # Selection box per prompt
        selection_box = Rectangle(width=2.5, height=1.2, color=color_select, stroke_width=2)
        selection_box.move_to(axes_pop.get_center())
        
        def get_sample(n=50):
            samples = []
            while len(samples) < n:
                x = np.random.uniform(0.1, 7.9)
                y = np.random.uniform(0.01, 0.4)
                if y < shark_fin(x):
                    samples.append(x)
            return np.array(samples)

        # Sampling points animation
        current_sample = get_sample(50)
        sample_dots = VGroup(*[
            Dot(axes_pop.c2p(s, np.random.uniform(0.01, shark_fin(s))), radius=0.03, color=color_select) 
            for s in current_sample
        ])
        
        self.play(Create(selection_box))
        self.play(FadeIn(sample_dots, lag_ratio=0.01))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color3)
        mean_val = np.mean(current_sample)
        
        # Fix Issue 38: Group and place in C1-C3 area
        mean_text = DecimalNumber(mean_val, num_decimal_places=2, include_sign=False, color=color_mean, mob_class=Text)
        mean_label = Text("Mean:", font_size=20, color=color_mean)
        mean_group = VGroup(mean_label, mean_text).arrange(RIGHT, buff=0.2)
        self.place_in_area(mean_group, "C1", "C3", scale_factor=0.7)
        
        # Points converge to mean per prompt
        self.play(
            sample_dots.animate.scale(0.1).move_to(mean_group.get_center()),
            FadeOut(selection_box)
        )
        self.play(Write(mean_group))
        self.remove(sample_dots)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(color4)
        result_dot = Dot(axes_samp.c2p(mean_val, 0), radius=0.05, color=color_mean)
        
        self.play(TransformFromCopy(mean_text, result_dot))
        self.play(FadeOut(mean_group))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(color5)
        
        # Fast repetition for 100 samples
        bins = np.linspace(0, 10, 21)
        bin_counts = np.zeros(len(bins)-1)
        
        # Record initial mean
        idx = np.digitize(mean_val, bins) - 1
        if 0 <= idx < len(bin_counts):
            bin_counts[idx] += 1

        for i in range(100):
            s = get_sample(50)
            m = np.mean(s)
            
            b_idx = np.digitize(m, bins) - 1
            if 0 <= b_idx < len(bin_counts):
                y_pos = bin_counts[b_idx] * 0.15 # Scale height of dots to fit
                bin_counts[b_idx] += 1
                new_dot = Dot(axes_samp.c2p(m, y_pos), radius=0.03, color=color_mean)
                self.add(new_dot)
                
                # Speed up as it goes
                self.wait(0.04 if i < 10 else 0.005)
        
        self.wait(2)
