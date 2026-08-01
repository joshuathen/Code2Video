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

class Section6Scene(TeachingScene):
    def construct(self):
        title_text = "Real-World Application: Signal Processing and Games"
        lecture_lines = [
            "Convolution filters noise in electronic signals and images.",
            "Game designers use it to calculate complex damage ranges.",
            "It's a powerful tool for understanding sums of random events."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # "Convolution filters noise in electronic signals and images."
        # Color: #00FFFF
        self.play(self.lecture[0].animate.set_color("#00FFFF"), run_time=1)
        
        # Noisy signal
        x_vals = np.linspace(-1.5, 1.5, 80)
        # Using a fixed seed for consistency
        rng = np.random.default_rng(42)
        noisy_pts = [np.array([x, 0.4 * np.sin(4 * x) + 0.2 * (rng.random() - 0.5), 0]) for x in x_vals]
        noisy_signal = VMobject(color="#00FFFF")
        noisy_signal.set_points_as_corners(noisy_pts)
        
        # Smooth signal
        smooth_pts = [np.array([x, 0.4 * np.sin(4 * x), 0]) for x in x_vals]
        smooth_signal = VMobject(color="#00FFFF")
        smooth_signal.set_points_as_corners(smooth_pts)
        
        # Position signals in area A2 to B5
        self.place_in_area(noisy_signal, 'A2', 'B5', scale_factor=1.0)
        self.place_in_area(smooth_signal, 'A2', 'B5', scale_factor=1.0)
        
        self.play(Create(noisy_signal))
        self.wait(1)
        self.play(Transform(noisy_signal, smooth_signal))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Game designers use it to calculate complex damage ranges."
        # Color: #FF69B4
        self.play(self.lecture[1].animate.set_color("#FF69B4"), run_time=1)
        
        # Display game dice 1-6 (#FFFFFF) and 1-4 (#D3D3D3)
        dice1 = VGroup(
            Square(side_length=0.8, color=WHITE, fill_opacity=0.2),
            Text("1-6", font_size=24, color=WHITE)
        )
        dice2 = VGroup(
            Square(side_length=0.8, color="#D3D3D3", fill_opacity=0.2),
            Text("1-4", font_size=24, color="#D3D3D3")
        )
        
        # Move dice1/2 to 'C3' and 'C4' (scale 0.8)
        self.place_at_grid(dice1, 'C3', scale_factor=0.8)
        self.place_at_grid(dice2, 'C4', scale_factor=0.8)
        
        # Resulting damage graph (#FF69B4)
        # Counts for sums 2..10: [1, 2, 3, 4, 4, 4, 3, 2, 1]
        counts = [1, 2, 3, 4, 4, 4, 3, 2, 1]
        bars = VGroup()
        for val in counts:
            bar = Rectangle(
                height=val * 0.25, 
                width=0.25, 
                fill_opacity=0.8, 
                fill_color="#FF69B4", 
                stroke_width=1,
                stroke_color=WHITE
            )
            bars.add(bar)
        bars.arrange(RIGHT, aligned_edge=DOWN, buff=0.1)
        
        # Labels for the graph
        graph_label = Text("Damage Distribution", font_size=18, color="#FF69B4")
        graph_vgroup = VGroup(bars, graph_label).arrange(DOWN, buff=0.2)
        
        # Expand 'graph_vgroup' to 'D2'-'F5' for better readability
        self.place_in_area(graph_vgroup, 'D2', 'F5', scale_factor=1.0)
        
        self.play(FadeIn(dice1), FadeIn(dice2))
        self.wait(0.5)
        self.play(FadeIn(graph_vgroup))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "It's a powerful tool for understanding sums of random events."
        # Keep lecture line 3 white (default) or highlight
        self.play(self.lecture[2].animate.set_color(WHITE), run_time=1)
        self.play(Indicate(self.lecture[2]))
        self.wait(2)
