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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Observation forces the wave function to collapse.",
            "The system chooses one definite state.",
            "Measurement determines the final outcome.",
            "Schrödinger's cat illustrates this measurement paradox.",
            "The box reveal forces reality to resolve."
        ]
        self.setup_layout("The Act of Measurement: Wave Function Collapse", lecture_lines)
        
        # Elements for animation
        # Wave function (sine wave)
        wave = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), x_range=[-2, 2], color=BLUE)
        self.place_in_area(wave, 'B4', 'D6', scale_factor=0.6)
        
        # Assets
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        box_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg")
        
        # Observer icon (using asset)
        observer = cat_icon
        self.place_at_grid(observer, 'E4', scale_factor=0.5)
        observer.set_opacity(0)
        
        # A collapsed state (a tall spike at box position)
        spike = Line(start=ORIGIN, end=UP*1.5, color=RED)
        self.place_at_grid(spike, 'D5') # Place spike at D5
        
        # box_icon placed at D5
        self.place_at_grid(box_icon, 'D5', scale_factor=0.3)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"), Create(wave))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"), observer.animate.set_opacity(1))
        self.add_label(observer, 'Observer', position='F4', color='#FFFFFF')
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"), FadeOut(wave), Create(spike))
        self.play(FadeIn(box_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF8000"))
        self.wait(1)
