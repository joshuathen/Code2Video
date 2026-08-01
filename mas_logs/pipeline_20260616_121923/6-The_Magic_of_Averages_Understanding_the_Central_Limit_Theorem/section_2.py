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
        # Initial layout setup with updated script
        lines = [
            'First, we select a random sample of thirty giraffes.',
            'Next, we calculate the average height for this group.',
            'This average becomes a single dot on our graph.',
            'We repeat this process multiple times to gather data.',
            'Watch as a thousand sample averages rain down below.'
        ]
        self.setup_layout("The Sampling Experiment", lines)

        # --- Scene Preparation ---
        # Number lines for population and sampling distribution
        top_axis = NumberLine(
            x_range=[0, 10, 1], 
            length=5, 
            include_numbers=True, 
            font_size=16, 
            color=WHITE,
            label_constructor=Text
        )
        bottom_axis = NumberLine(
            x_range=[0, 10, 1], 
            length=5, 
            include_numbers=True, 
            font_size=16, 
            color=WHITE,
            label_constructor=Text
        )
        
        self.place_in_area(top_axis, 'B1', 'B6')
        self.place_in_area(bottom_axis, 'F1', 'F6')
        
        pop_label = Text("Population Heights", font_size=14).next_to(top_axis, UP, buff=0.1)
        dist_label = Text("Sample Means Distribution", font_size=14).next_to(bottom_axis, UP, buff=0.1)
        
        # Generate Population
        np.random.seed(42)
        population_vals = np.random.normal(5, 1.2, 100)
        population_vals = np.clip(population_vals, 0.5, 9.5)
        pop_dots = VGroup(*[
            Dot(top_axis.n2p(v) + UP * np.random.uniform(0.1, 0.4), radius=0.03, color="#888888") 
            for v in population_vals
        ])

        self.add(top_axis, bottom_axis, pop_label, dist_label, pop_dots)
        self.wait(1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Select 30 random dots
        sample_indices = np.random.choice(range(100), 30, replace=False)
        sample_dots = VGroup(*[pop_dots[i] for i in sample_indices])
        
        # Display sample size n=30 (Issue 42 fix)
        n_label = Text("Sample size: n = 30", font_size=20, color="#FFD700")
        self.place_in_area(n_label, 'C3', 'C4', scale_factor=0.8)
        
        self.play(
            sample_dots.animate.set_color("#00FF00").scale(1.5),
            Write(n_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Calculation Box (Issue 43 fix)
        calc_box = RoundedRectangle(corner_radius=0.1, width=3.0, height=0.6, color=WHITE)
        self.place_in_area(calc_box, 'D3', 'D4', scale_factor=0.85)
        calc_text = Text("Calculate Average", font_size=16, color=WHITE).move_to(calc_box.get_center())
        calculation_group = VGroup(calc_box, calc_text)
        
        # Animation: Merge green dots into one yellow dot
        current_mean = np.mean([population_vals[i] for i in sample_indices])
        mean_dot = Dot(calc_box.get_center(), color="#FFFF00", radius=0.08)
        
        self.play(Create(calculation_group))
        self.play(
            sample_dots.copy().animate.move_to(calc_box.get_center()).scale(0.1).set_opacity(0),
            FadeIn(mean_dot),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Target position on bottom axis
        target_pos = bottom_axis.n2p(current_mean)
        
        self.play(
            mean_dot.animate.move_to(target_pos),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        
        # Hide calculation box and n_label for clarity during repeats
        self.play(FadeOut(calculation_group), FadeOut(n_label))
        
        # Repeat sampling 5 times quickly to show the process
        collected_dots = VGroup(mean_dot)
        for _ in range(5):
            new_indices = np.random.choice(range(100), 30, replace=False)
            new_mean = np.mean([population_vals[i] for i in new_indices])
            new_dot = Dot(bottom_axis.n2p(new_mean), color="#FFFF00", radius=0.06)
            
            # Temporary highlight of sampled population dots
            temp_sample = VGroup(*[pop_dots[i] for i in new_indices])
            self.play(
                temp_sample.animate(run_time=0.2).set_color("#00FF00"),
                rate_func=linear
            )
            # Create dot and move to axis
            new_dot.move_to(self.grid["D3"])
            self.play(FadeIn(new_dot, run_time=0.2))
            self.play(new_dot.animate(run_time=0.3).move_to(bottom_axis.n2p(new_mean)))
            self.play(temp_sample.animate(run_time=0.1).set_color("#888888"), rate_func=linear)
            collected_dots.add(new_dot)

        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        
        # Final "Rain" of dots
        rain_dots = VGroup()
        for _ in range(200): # Using 200 for performance while looking like a lot
            # CLT distribution is narrower: std_dev / sqrt(n)
            m = np.random.normal(5, 1.2 / np.sqrt(30))
            # Keep within bounds
            m = np.clip(m, 0.5, 9.5)
            # Stacking effect: random jitter in y
            jitter_y = np.random.uniform(0, 0.6)
            d = Dot(bottom_axis.n2p(m) + UP * jitter_y, color="#FFFF00", radius=0.04, fill_opacity=0.6)
            rain_dots.add(d)
        
        self.play(
            LaggedStart(
                *[FadeIn(d, shift=DOWN*0.5) for d in rain_dots],
                lag_ratio=0.01,
                run_time=4
            )
        )
        
        self.wait(2)
