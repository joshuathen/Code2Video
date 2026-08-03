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

class Section5Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "The Golden Rule: Sample Size Matters"
        lecture_lines = [
            "The theorem works best when sample sizes are large.",
            "Generally, a sample size of thirty or more is ideal.",
            "Larger samples create a narrower, more precise bell curve.",
            "Small samples result in wider, less reliable distributions.",
            "Precision increases as we include more data per sample."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Two separate axes appear: 'Small Sample' on the left, 'Large Sample' on the right.
        self.lecture[0].set_color("#FFFF00")
        
        axes_small = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.6, 0.2],
            x_length=2.5,
            y_length=1.5,
            axis_config={"include_tip": False, "include_numbers": False, "color": BLUE_D},
        )
        self.place_in_area(axes_small, "B1", "D3")
        
        label_small = Text("Small Sample", font_size=18, color=WHITE)
        self.place_at_grid(label_small, "A2")
        
        axes_large = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 1.2, 0.4],
            x_length=2.5,
            y_length=1.5,
            axis_config={"include_tip": False, "include_numbers": False, "color": BLUE_D},
        )
        self.place_in_area(axes_large, "B4", "D6")
        
        label_large = Text("Large Sample", font_size=18, color=WHITE)
        self.place_at_grid(label_large, "A5")

        self.play(Create(axes_small), Create(axes_large), Write(label_small), Write(label_large))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Generally, a sample size of thirty or more is ideal.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        n_small = MathTex("n = 5", font_size=22, color="#FFA500")
        self.place_at_grid(n_small, "E2") # Resolved Issue 30: Avoid overlap with axes_small
        
        n_large = MathTex("n = 50", font_size=22, color="#00FF00")
        self.place_at_grid(n_large, "E5") # Resolved Issue 31: Avoid overlap with axes_large
        
        self.play(Write(n_small), Write(n_large))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Larger samples create a narrower, more precise bell curve.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FF00")
        
        def normal_pdf(x, mu, sigma):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            
        curve_large = axes_large.plot(
            lambda x: normal_pdf(x, 0, 0.4),
            color="#00FF00",
            x_range=[-3.5, 3.5]
        )
        
        self.play(Create(curve_large))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Small samples result in wider, less reliable distributions.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFA500")
        
        curve_small = axes_small.plot(
            lambda x: normal_pdf(x, 0, 1.0),
            color="#FFA500",
            x_range=[-3.5, 3.5]
        )
        
        self.play(Create(curve_small))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Precision increases as we include more data per sample.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")
        
        summary_text = Text("Size n increases, Spread decreases", font_size=20, color=WHITE)
        self.place_in_area(summary_text, "F1", "F6")
        
        self.play(Flash(curve_large, color="#00FF00", flash_radius=1.2, line_length=0.3))
        self.play(Write(summary_text))
        self.wait(2)
