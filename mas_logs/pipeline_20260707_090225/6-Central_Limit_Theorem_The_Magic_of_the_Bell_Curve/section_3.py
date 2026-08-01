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
        lines = [
            'We repeatedly take random samples of thirty squirrels each.',
            "Instead of plotting individuals, we calculate each group's average.",
            'These averages are recorded on a new distribution graph.',
            'Watch as each new average falls into its slot.',
            'A familiar shape slowly begins to emerge from chaos.'
        ]
        self.setup_layout("The Experiment: Gathering the Averages", lines)

        # Define line colors
        colors = ["#00FF00", "#FFFF00", "#00FFFF", "#FFA500", "#FF00FF"]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        
        # Population Axes (A2-C6 area) - Using label_constructor=Text to avoid LaTeX dependency
        pop_axes = Axes(
            x_range=[0, 10, 2], y_range=[0, 1, 0.5], 
            x_length=4.5, y_length=1.5, 
            tips=False, axis_config={
                "color": GREY_B, 
                "include_numbers": True, 
                "font_size": 14,
                "label_constructor": Text
            }
        )
        # Fix Issue 36: Move pop_axes down and scale slightly
        self.place_in_area(pop_axes, "A2", "C6", scale_factor=0.9)
        pop_label = Text("Chaotic Population Distribution", font_size=16, color=GREY_B).next_to(pop_axes, UP, buff=0.1)
        
        # Create messy bimodal data
        np.random.seed(123)
        pop_data = np.concatenate([
            np.random.normal(3, 0.6, 100), 
            np.random.normal(7.5, 0.4, 100),
            np.random.uniform(1, 9, 50)
        ])
        pop_data = np.clip(pop_data, 0, 10)
        
        pop_dots = VGroup(*[
            Dot(pop_axes.c2p(x, np.random.uniform(0.1, 0.8)), radius=0.03, color=BLUE_D, fill_opacity=0.3)
            for x in pop_data
        ])

        sampling_box = Rectangle(width=0.8, height=0.6, color="#00FF00", stroke_width=2)
        # Fix Issue 37: Center sampling_box at B4
        self.place_at_grid(sampling_box, "B4", scale_factor=1.0)
        
        self.play(Create(pop_axes), FadeIn(pop_label), FadeIn(pop_dots))
        self.play(Create(sampling_box))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        
        # Group of captured dots
        sample_group = VGroup(*[
            Dot(radius=0.03, color="#00FF00").move_to(
                sampling_box.get_center() + np.array([np.random.uniform(-0.3, 0.3), np.random.uniform(-0.2, 0.2), 0])
            ) for _ in range(30)
        ])
        
        # Construct average symbol (x-bar) manually to avoid MathTex/LaTeX
        avg_x_text = Text("x", color=colors[1])
        avg_x_bar = Line(color=colors[1], stroke_width=2).match_width(avg_x_text).scale(0.8).next_to(avg_x_text, UP, buff=0.05)
        avg_symbol = VGroup(avg_x_text, avg_x_bar).scale(0.8)
        avg_symbol.next_to(sampling_box, RIGHT, buff=0.2)

        self.play(FadeIn(sample_group))
        self.play(Write(avg_symbol))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        
        # Sampling Distribution Axes (D3-F6 area) - Using label_constructor=Text to avoid LaTeX dependency
        sample_axes = Axes(
            x_range=[0, 10, 2], y_range=[0, 120, 30], 
            x_length=4.5, y_length=1.5, 
            tips=False, axis_config={
                "color": GREY_B, 
                "include_numbers": True, 
                "font_size": 14,
                "label_constructor": Text
            }
        )
        # Fix Issue 38: Move sample_axes down to D3-F6 and scale
        self.place_in_area(sample_axes, "D3", "F6", scale_factor=0.9)
        sample_label_text = Text("Sampling Distribution of Means", font_size=16, color=WHITE).next_to(sample_axes, UP, buff=0.1)
        
        self.play(Create(sample_axes), FadeIn(sample_label_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(colors[3])
        
        # Histogram buildup setup
        num_bins = 20
        bin_width = 10 / num_bins
        bins = np.linspace(0, 10, num_bins + 1)
        counts = np.zeros(num_bins)
        
        bars = VGroup(*[
            Rectangle(
                width=sample_axes.x_axis.get_unit_size() * bin_width * 0.9,
                height=0.01,
                fill_color=colors[3],
                fill_opacity=0.7,
                stroke_width=0.5
            ).move_to(sample_axes.c2p(bins[i] + bin_width/2, 0), aligned_edge=DOWN)
            for i in range(num_bins)
        ])
        self.add(bars)

        # Sampling logic
        def get_sample_mean():
            indices = np.random.choice(len(pop_data), 30)
            return np.mean(pop_data[indices])

        # Animate first 5 samples slowly
        for _ in range(5):
            m = get_sample_mean()
            # Flash sampling box to indicate action
            self.play(sampling_box.animate.set_stroke(color=YELLOW, width=4), run_time=0.1)
            self.play(sampling_box.animate.set_stroke(color="#00FF00", width=2), run_time=0.1)
            
            # Dot represents the average being calculated and "falling"
            dot = Dot(sampling_box.get_center(), radius=0.05, color=colors[3])
            target_pos = sample_axes.c2p(m, 0)
            
            self.play(dot.animate.move_to(target_pos), run_time=0.4)
            
            # Find correct bin and update bar height
            bin_idx = np.digitize(m, bins) - 1
            if 0 <= bin_idx < num_bins:
                counts[bin_idx] += 1
                new_height = sample_axes.y_axis.get_unit_size() * counts[bin_idx]
                self.play(
                    bars[bin_idx].animate.stretch_to_fit_height(new_height, about_edge=DOWN),
                    FadeOut(dot),
                    run_time=0.2
                )

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        
        # Rapidly add many more samples to show the shape emerging
        total_additional = 400
        batch_size = 40
        num_batches = total_additional // batch_size
        
        for _ in range(num_batches):
            for _ in range(batch_size):
                m = get_sample_mean()
                bin_idx = np.digitize(m, bins) - 1
                if 0 <= bin_idx < num_bins:
                    counts[bin_idx] += 1
            
            # Batch update the bar heights
            batch_anims = []
            for i in range(num_bins):
                new_height = sample_axes.y_axis.get_unit_size() * counts[i]
                if new_height > 0:
                    batch_anims.append(bars[i].animate.stretch_to_fit_height(new_height, about_edge=DOWN))
            self.play(*batch_anims, run_time=0.1)

        # Highlight emerging shape with a normal distribution curve
        pop_mean = np.mean(pop_data)
        pop_std = np.std(pop_data)
        sample_std = pop_std / np.sqrt(30)
        
        # Total samples visualized is 5 + 400 = 405
        norm_curve = sample_axes.plot(
            lambda x: 405 * bin_width * (1/(sample_std * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x-pop_mean)/sample_std)**2),
            color=colors[4], stroke_width=3
        )
        self.play(Create(norm_curve))
        self.wait(2)
