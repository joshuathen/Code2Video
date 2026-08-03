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
        self.setup_layout(
            "Prerequisite Knowledge: Populations vs. Samples",
            [
                "A population represents every individual in a giant group.",
                "We often only study a small subset called a sample.",
                "The sample mean is the average of that small group."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # A large blue (#0000FF) circle labeled 'Population' appears with many dots inside.
        self.lecture[0].set_color(BLUE)
        
        population_circle = Circle(radius=1.3, color="#0000FF", fill_opacity=0.1)
        self.place_in_area(population_circle, 'A1', 'D3')
        
        population_label = Text("Population", font_size=20, color="#0000FF")
        self.place_at_grid(population_label, 'E2')
        
        # Create many dots inside the population circle
        np.random.seed(42) # For consistent dot placement
        dots = VGroup()
        for _ in range(40):
            # Generate random points within the circle
            angle = np.random.uniform(0, 2 * PI)
            r = np.sqrt(np.random.uniform(0, 1)) * 1.1 # slightly smaller than radius 1.3
            point = population_circle.get_center() + np.array([r * np.cos(angle), r * np.sin(angle), 0])
            dots.add(Dot(point=point, radius=0.04, color=WHITE))
        
        self.play(
            Create(population_circle),
            Write(population_label),
            FadeIn(dots, lag_ratio=0.05),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A smaller green (#00FF00) circle labeled 'Sample' appears to the right.
        self.lecture[1].set_color(GREEN)
        
        sample_circle = Circle(radius=0.8, color="#00FF00", fill_opacity=0.1)
        self.place_in_area(sample_circle, 'B4', 'D6')
        
        sample_label = Text("Sample", font_size=20, color="#00FF00")
        self.place_at_grid(sample_label, 'E5')
        
        self.play(
            Create(sample_circle),
            Write(sample_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Five dots from the Population circle move into the Sample circle via white (#FFFFFF) arrows.
        self.lecture[2].set_color(YELLOW)
        
        # Pick 5 dots to move
        dots_to_move = dots[:5]
        
        arrows = VGroup()
        target_positions = []
        for i in range(5):
            # Target positions inside the sample circle
            angle = np.random.uniform(0, 2 * PI)
            r = np.sqrt(np.random.uniform(0, 1)) * 0.6 # slightly smaller than sample radius 0.8
            target_pos = sample_circle.get_center() + np.array([r * np.cos(angle), r * np.sin(angle), 0])
            target_positions.append(target_pos)
            
            # Create arrows from original positions to targets
            arrow = Arrow(
                start=dots_to_move[i].get_center(),
                end=target_pos,
                color="#FFFFFF",
                buff=0.1,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.15
            )
            arrows.add(arrow)
            
        self.play(
            Create(arrows),
            run_time=1
        )
        
        # Calculate mean for the label (visual representation)
        mean_label = MathTex(r"\bar{x} = \frac{\sum x_i}{n}", font_size=24, color=YELLOW)
        self.place_in_area(mean_label, 'F4', 'F6')

        self.play(
            *[dots_to_move[i].animate.move_to(target_positions[i]) for i in range(5)],
            FadeOut(arrows),
            Write(mean_label),
            run_time=2
        )
        self.wait(2)
