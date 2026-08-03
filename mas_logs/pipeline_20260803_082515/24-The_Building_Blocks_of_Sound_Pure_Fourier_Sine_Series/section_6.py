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
        # Setup layout
        title_text = "Real-World Application: Audio Compression"
        lecture_lines = [
            "Computers store these harmonic recipes instead of raw data.",
            "This \"shorthand\" allows for efficient digital audio compression.",
            "Fourier series turn complex sounds into simple, manageable lists."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for lecture lines
        c1 = "#FFFF00"  # Yellow for recipe
        c2 = "#00FF00"  # Green for reconstruction
        c3 = "#FFFFFF"  # White for labels

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(c1))

        # Recipe List: b1, b3, b5, b7, b9
        recipe_title = Text("Recipe [bn]", font_size=24, color=c1)
        # Position per Issue 40: B1
        self.place_at_grid(recipe_title, "B1", scale_factor=0.8)
        
        bn_values = [
            MathTex("b_1 = 1.00", color=c1),
            MathTex("b_3 = 0.33", color=c1),
            MathTex("b_5 = 0.20", color=c1),
            MathTex("b_7 = 0.14", color=c1),
            MathTex("b_9 = 0.11", color=c1)
        ]
        recipe_list = VGroup(*bn_values).arrange(DOWN, aligned_edge=LEFT)
        # Position per Issue 40: C1 to E2
        self.place_in_area(recipe_list, "C1", "E2", scale_factor=0.7)

        self.play(Write(recipe_title))
        self.play(LaggedStart(*[Write(val) for val in bn_values], lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(c2))

        # Create Axes for wave reconstruction
        axes = Axes(
            x_range=[0, 4 * PI, PI],
            y_range=[-1.5, 1.5, 1],
            axis_config={"color": GREY},
            tips=False
        )
        # Position per Issue 40: B3 to E6
        self.place_in_area(axes, "B3", "E6", scale_factor=0.6)

        # Function for square wave reconstruction (5 harmonics)
        # 4/pi * sum sin(nx)/n for odd n
        def square_wave_recon(x):
            return (4/PI) * (
                np.sin(x) + 
                (1/3) * np.sin(3*x) + 
                (1/5) * np.sin(5*x) + 
                (1/7) * np.sin(7*x) + 
                (1/9) * np.sin(9*x)
            )

        wave_graph = axes.plot(square_wave_recon, color=c2)

        # Transform the recipe into the wave
        self.play(
            ReplacementTransform(recipe_list, wave_graph),
            FadeOut(recipe_title),
            Create(axes)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(c3))

        label_compression = Text("Audio Compression", font_size=24, color=c3)
        label_mp3 = Text("MP3 Standard", font_size=24, color=c3)

        # Position per Issue 41: F3 and F5
        self.place_at_grid(label_compression, "F3", scale_factor=0.8)
        self.place_at_grid(label_mp3, "F5", scale_factor=0.8)

        # Asset integration per Issue 27
        computer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color=WHITE)
        self.place_at_grid(computer_icon, "F1", scale_factor=0.6)

        self.play(
            Write(label_compression),
            Write(label_mp3),
            DrawBorderThenFill(computer_icon)
        )
        self.wait(2)
