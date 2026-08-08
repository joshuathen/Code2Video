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
        # Set seed for reproducible monster scatter
        np.random.seed(42)
        
        self.setup_layout("Prerequisite Knowledge: Populations vs. Samples", [
            "The entire forest population has a mean and spread.",
            "Zog captures a small random group called a sample.",
            "This sample's average height is the sample mean."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Highlight current lecture line
        self.lecture[0].set_color(WHITE)
        
        # Monsters (forest population) represented by small circular blobs
        monsters = VGroup(*[
            Circle(radius=0.07, color=GREY_B, fill_opacity=0.7, stroke_width=1) 
            for _ in range(40)
        ])
        
        # Scatter monsters randomly within the forest area (roughly B2 to E5)
        for m in monsters:
            m.move_to([
                np.random.uniform(1.0, 5.0),
                np.random.uniform(-2.5, 1.5),
                0
            ])

        # Large dashed white circle around all monsters (Population)
        pop_circle = DashedVMobject(Circle(radius=2.7, color=WHITE))
        self.place_in_area(pop_circle, "A1", "F6")
        
        # Label 'Population (N)' at top - Position fixed per Issue 26
        pop_label = Text("Population (N)", font_size=24, color=WHITE)
        self.place_at_grid(pop_label, "A2", scale_factor=0.8)
        
        # Mean (mu) and Spread (sigma) symbols for the population
        mu_sym = MathTex("\\mu", color=WHITE, font_size=44)
        sigma_sym = MathTex("\\sigma", color=WHITE, font_size=44)
        
        # Position mu at E3 to be side-by-side with x-bar later
        self.place_at_grid(mu_sym, "E3")
        self.place_at_grid(sigma_sym, "E2")
        
        self.play(
            FadeIn(monsters),
            Create(pop_circle),
            Write(pop_label),
            run_time=2
        )
        self.play(Write(mu_sym), Write(sigma_sym))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update lecture line colors
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        
        # Solid green rectangle 'Net' around 5 monsters
        net = Rectangle(width=2.2, height=1.6, color=GREEN, stroke_width=4)
        self.place_in_area(net, "C3", "D4") 
        
        # Label 'Sample (n=5)' - Position fixed per Issue 24
        sample_label = Text("Sample (n=5)", font_size=24, color=GREEN)
        self.place_at_grid(sample_label, "B3", scale_factor=0.8)
        
        # Pick 5 monsters to move into the net
        sample_indices = [5, 12, 19, 26, 33]
        sample_group = VGroup(*[monsters[i] for i in sample_indices])
        
        sample_move_anims = []
        for i, m in enumerate(sample_group):
            # Arrange monsters nicely inside the net
            angle = i * (2 * PI / 5)
            offset = np.array([0.45 * np.cos(angle), 0.35 * np.sin(angle), 0])
            sample_move_anims.append(m.animate.move_to(net.get_center() + offset).set_color(GREEN))
            
        self.play(
            Create(net),
            Write(sample_label),
            *sample_move_anims,
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Update lecture line colors
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
