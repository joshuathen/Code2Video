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
        # Data from storyboard
        title = "Summary & Real-World Impact"
        lecture_lines = [
            "Fourier analysis decomposes signals into their constituent DNA.",
            "Prisms of math reveal the hidden frequencies inside data.",
            "This compression enables modern music and digital imaging."
        ]
        
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_LINE1 = "#00FFFF" # Cyan for complex wave
        COLOR_LINE2 = "#FFFF00" # Yellow for pure sine waves
        COLOR_LINE3 = "#FFFFFF" # White for icons
        PRISM_COLOR = "#AAAAAA" # Gray for prism

        # === Animation for Lecture Line 1 ===
        # "Fourier analysis decomposes signals into their constituent DNA."
        self.lecture[0].set_color(COLOR_LINE1)
        
        # Complex wave: Sum of 3 sines
        complex_wave = FunctionGraph(
            lambda x: 0.6 * np.sin(x*2) + 0.3 * np.sin(x*5) + 0.1 * np.sin(x*10),
            x_range=[-1.5, 1.5],
            color=COLOR_LINE1
        )
        # Resolved Issue 36: Move complex_wave to C2
        self.place_at_grid(complex_wave, "C2", scale_factor=0.6)
        
        # Resolved Issue 22: Use SVGMobject for prism
        # Use existing asset reference
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        prism.set_color(PRISM_COLOR)
        # Resolved Issue 37: Move prism to C3
        self.place_at_grid(prism, "C3", scale_factor=1.0)
        
        self.play(FadeIn(prism))
        self.play(Create(complex_wave))
        
        # Wave enters prism and disappears
        # Move toward C3 (the prism's location)
        self.play(
            complex_wave.animate.move_to(self.grid["C3"]).scale(0.1).set_stroke(opacity=0),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # "Prisms of math reveal the hidden frequencies inside data."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_LINE2)
        
        # Pure sine waves exiting the prism at different angles
        sine1 = FunctionGraph(lambda x: 0.4 * np.sin(x*2), x_range=[-0.8, 0.8], color=COLOR_LINE2)
        sine2 = FunctionGraph(lambda x: 0.3 * np.sin(x*5), x_range=[-0.8, 0.8], color=COLOR_LINE2)
        sine3 = FunctionGraph(lambda x: 0.2 * np.sin(x*10), x_range=[-0.8, 0.8], color=COLOR_LINE2)
        
        # Start them all small inside the prism at C3
        for s in [sine1, sine2, sine3]:
            s.move_to(self.grid["C3"]).scale(0.1).set_stroke(opacity=0)
            
        # Animate them exiting to different grid points on the right
        self.play(
            sine1.animate.move_to(self.grid["A6"]).scale(8.0).set_stroke(opacity=1),
            sine2.animate.move_to(self.grid["C6"]).scale(8.0).set_stroke(opacity=1),
            sine3.animate.move_to(self.grid["E6"]).scale(8.0).set_stroke(opacity=1),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "This compression enables modern music and digital imaging."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_LINE3)
        
        # Icons for MP3 and JPEG representing data compression
        mp3_label = Text("MP3", font_size=24, color=WHITE)
        mp3_rect = RoundedRectangle(corner_radius=0.1, height=0.6, width=1.2, color=WHITE)
        mp3_icon = VGroup(mp3_rect, mp3_label)
        
        jpeg_label = Text("JPEG", font_size=24, color=WHITE)
        jpeg_rect = RoundedRectangle(corner_radius=0.1, height=0.6, width=1.2, color=WHITE)
        jpeg_icon = VGroup(jpeg_rect, jpeg_label)
        
        # Resolved Issue 38: Move icons to B4 and D4
        self.place_at_grid(mp3_icon, "B4", scale_factor=1.0)
        self.place_at_grid(jpeg_icon, "D4", scale_factor=1.0)
        
        self.play(
            FadeIn(mp3_icon, shift=RIGHT),
            FadeIn(jpeg_icon, shift=RIGHT)
        )
        self.wait(3)
