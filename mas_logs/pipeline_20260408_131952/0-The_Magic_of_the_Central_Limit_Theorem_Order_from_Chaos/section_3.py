from manim import *
import random

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
        # 1. Setup layout
        lecture_lines = [
            "Let's pick random groups of thirty creatures each.",
            "Calculate each group's average height and record it.",
            "We repeat this process hundreds of times."
        ]
        self.setup_layout("The Sampling Experiment", lecture_lines)
        
        # Color definitions for alignment
        color_1 = BLUE_B
        color_2 = GREEN_B
        color_3 = GOLD_A
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_1)
        
        # Forest population (collection of creatures)
        population = VGroup()
        random.seed(42) # Deterministic for layout
        for _ in range(80):
            # Creatures as small dots
            dot = Dot(radius=0.03, color=GRAY_C)
            # Relative spread for the area
            dot.shift(RIGHT * random.uniform(-2.2, 2.2) + UP * random.uniform(-0.8, 0.8))
            population.add(dot)
        
        # Fix Issue 45: Adjust area to 'C5' to avoid overlap with 'B6'
        self.place_in_area(population, "A1", "C5")
        self.add(population)
        
        # Select 30 creatures (sample size n=30)
        sample_indices = random.sample(range(len(population)), 30)
        sample_creatures = VGroup(*[population[i] for i in sample_indices])
        
        label_n = Text("Sample Size n=30", font_size=24, color=color_1)
        # Fix Issue 46: Scale down the label at B6
        self.place_at_grid(label_n, "B6", scale_factor=0.6)
        
        self.play(
            sample_creatures.animate.set_color(color_1).scale(1.4),
            Write(label_n),
            run_time=1.5
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_2)
        
        # Sampling Axis (where means land)
        sampling_axis = NumberLine(
            x_range=[0, 10, 1],
            length=5,
            include_ticks=True,
            color=WHITE,
            stroke_width=2
        )
        self.place_in_area(sampling_axis, "E1", "F6")
        axis_label = Text("Sample Mean Value", font_size=18, color=WHITE)
        # Fix Issue 47: Center axis label horizontally over the axis area
        self.place_in_area(axis_label, "F1", "F6", scale_factor=0.8)
        
        self.play(Create(sampling_axis), FadeIn(axis_label))
        
        # Calculate visual mean
        mean_ball = Dot(radius=0.1, color=color_2)
        mean_ball.move_to(sample_creatures.get_center())
        
        # Landing position on axis
        # Assuming the population mean is around center (5.0)
        mean_val = 5.2
        landing_pos = sampling_axis.n2p(mean_val)
        
        # Merge sample into mean ball and drop
        self.play(
            sample_creatures.animate.move_to(mean_ball.get_center()).scale(0.1).set_opacity(0),
            FadeIn(mean_ball),
            run_time=1.2
        )
        self.play(
            mean_ball.animate.move_to(landing_pos),
            rate_func=rush_into,
            run_time=1.0
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color_3)
        
        # Fast repetition loop
        # Store collected means to show progress
        collected_means = VGroup(mean_ball)
        
        # Perform several visible iterations
        for i in range(6):
            # Logic: Select new random group, show them briefly, drop ball
            new_indices = random.sample(range(len(population)), 30)
            new_sample = VGroup(*[population[idx] for idx in new_indices])
            
            # Reset visual of previously used dots (if any were the same)
            # and highlight new ones
            val = random.gauss(5, 0.6)
            t_pos = sampling_axis.n2p(val)
            
            # Move dots back to normal and hide color for others
            # But simpler: just flash the new sample
            m_ball = Dot(radius=0.08, color=color_3)
            m_ball.move_to(new_sample.get_center())
            
            self.play(
                new_sample.animate(run_time=0.2).set_color(color_3),
                FadeIn(m_ball, run_time=0.1)
            )
            self.play(
                m_ball.animate(run_time=0.3).move_to(t_pos),
                new_sample.animate(run_time=0.2).set_color(GRAY_C)
            )
            collected_means.add(m_ball)
            
        # Final flourish: drop many balls quickly to represent "hundreds"
        extra_balls = VGroup()
        for _ in range(50):
            val = random.gauss(5, 0.6)
            eb = Dot(radius=0.06, color=color_3, fill_opacity=0.6)
            # Stack dots slightly to show density
            stack_height = random.uniform(0.05, 0.4)
            eb.move_to(sampling_axis.n2p(val) + UP * stack_height)
            extra_balls.add(eb)
            
        self.play(
            LaggedStartMap(FadeIn, extra_balls, lag_ratio=0.02, run_time=2),
            label_n.animate.set_color(WHITE) # Finished highlighting
        )
        
        self.wait(3)
