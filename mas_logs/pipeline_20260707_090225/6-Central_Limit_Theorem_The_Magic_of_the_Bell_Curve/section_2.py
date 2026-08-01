from manim import *
import numpy as np
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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Kit: Populations and Samples", [
            "Let's define our population: a bag of diverse marbles.",
            "A sample is just a small scoop from that bag.",
            "The sample mean represents the average of that scoop."
        ])

        # Assets
        marble_asset = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/marbles.svg"

        # === Animation for Lecture Line 1 ===
        # Use light, distinguishable hexadecimal colors. Match color with corresponding elements.
        self.lecture[0].set_color(WHITE)
        
        # Population Container (Issue 34/48 Fix: B2 to E5)
        self.container = RoundedRectangle(
            width=4.0, height=3.5, corner_radius=0.3, color="#FFFFFF", stroke_width=3
        )
        self.place_in_area(self.container, "B2", "E5")
        
        # Population Label (Issue 34/48 Fix: A2, scale 0.7)
        self.pop_label = Text("Population", font_size=24, color="#FFFFFF")
        self.place_at_grid(self.pop_label, "A2", scale_factor=0.7)

        # Population Marbles (Issue 27/48 Fix: Asset integration)
        marbles = VGroup()
        colors = [RED_A, BLUE_A, YELLOW_A, GREEN_A, PINK, ORANGE, PURPLE_A, WHITE]
        random.seed(42)
        
        # Determine bounds based on container area (B2 to E5)
        for _ in range(100):
            # Coordinates range roughly B2(1.5, 1.2) to E5(4.5, -1.8)
            rx = random.uniform(1.7, 4.3)
            ry = random.uniform(-1.6, 1.0)
            dot = SVGMobject(marble_asset).scale(0.12).set_color(random.choice(colors))
            dot.move_to([rx, ry, 0])
            marbles.add(dot)

        self.play(Create(self.container), Write(self.pop_label))
        self.play(FadeIn(marbles, lag_ratio=0.005), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00") # Green
        
        # Selection of 5 marbles
        sample_indices = [15, 32, 45, 78, 92]
        sample_marbles = VGroup(*[marbles[i] for i in sample_indices])
        
        # Green scoop selection
        scoop_circle = Circle(radius=0.6, color="#00FF00", stroke_width=4)
        scoop_circle.move_to(sample_marbles.get_center())

        # Sample Label (Issue 35/48 Fix: F1, scale 0.8)
        self.sample_label = Text("Sample", font_size=20, color="#00FF00")
        self.place_at_grid(self.sample_label, "F1", scale_factor=0.8)

        self.play(Create(scoop_circle))
        self.play(sample_marbles.animate.set_color("#00FF00").scale(1.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00") # Yellow
        
        # Mean calculation display (Issue 33/48 Fix: F2-F4 for label, F5 for value)
        self.mean_label = Text("Sample Mean:", font_size=22, color="#FFFF00")
        self.place_in_area(self.mean_label, "F2", "F4", scale_factor=0.8)
        
        self.mean_val = DecimalNumber(0.0, num_decimal_places=1, color="#FFFF00", mob_class=Text)
        self.place_at_grid(self.mean_val, "F5", scale_factor=0.8)

        # Move the selected sample to the side area (Issue 48 asset integration)
        sample_move_group = VGroup(scoop_circle, sample_marbles)
        target_pos = self.grid["F1"] + UP * 0.7 # Move group above its label
        
        self.play(
            sample_move_group.animate.move_to(target_pos).scale(0.8),
            FadeIn(self.sample_label),
            run_time=2
        )
        
        # Final display of result
        self.play(
            Write(self.mean_label),
            ChangeDecimalToValue(self.mean_val, 14.8),
            run_time=1.5
        )
        self.wait(2)
