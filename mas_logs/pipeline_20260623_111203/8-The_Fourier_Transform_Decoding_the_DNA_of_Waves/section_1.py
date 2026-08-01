from manim import *
import numpy as np

class Section1Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Tex(title_text, font_size=32, color=WHITE).to_edge(UP)
        self.add(self.title)
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=32, color=WHITE).to_edge(UP)
        self.add(self.title)
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=32, color=WHITE).to_edge(UP)
        self.add(self.title)
        # Background and Title
        self.camera.background_color = "#000000"
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=32, color=WHITE).to_edge(UP)
        self.add(self.title)
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=32, color=WHITE).to_edge(UP)
        self.add(self.title)
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title)

        # Left-side lecture content
        lecture_texts = [Text(line, font_size=24, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.lecture.to_edge(LEFT, buff=1.0)
        
        # Define grid for positioning elements on the right side
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Offset grid to the right half of the screen
                x = 1.5 + j * 0.8
                y = 2.0 - i * 0.8
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def construct(self):
        # Configuration for "The Fourier Transform: Decoding the DNA of Waves"
        lecture_content = [
            "1. Time vs. Frequency",
            "2. Periodic Waveforms",
            "3. Mathematical Decomposition"
        ]
        
        self.setup_layout("The Fourier Transform", lecture_content)

        # Create a sample signal (Sine Wave)
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=4,
            y_length=3,
            axis_config={"color": BLUE_D, "include_tip": False}
        )
        
        sine_graph = axes.plot(lambda x: np.sin(2 * PI * x), color=YELLOW)
        wave_group = VGroup(axes, sine_graph)
        
        # Place graph using the grid system
        self.place_at_grid(wave_group, "C2", scale_factor=0.7)

        # Animations
        self.play(
            Write(self.title),
            run_time=1
        )
        
        self.play(
            FadeIn(self.lecture, shift=RIGHT),
            run_time=1.5
        )
        
        self.play(
            Create(axes),
            Create(sine_graph),
            run_time=2
        )
        
        # Highlight transformation concept
        transform_label = Text("Time Domain", font_size=20).next_to(axes, DOWN)
        self.play(Write(transform_label))
        
        self.wait(3)

if __name__ == "__main__":
    # Command to render: manim -pql test_section_1.py Section1Scene
    pass