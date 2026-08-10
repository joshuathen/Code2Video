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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Average Speed Paradox", [
            "Average speed is total distance over total time.",
            "Cheetah covers 100 meters in 5 seconds.",
            "Average speed is 20 meters per second.",
            "But this hides the cheetah's actual movement.",
            "Did it sprint or pause mid-hunt?"
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        # Load asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg", color="#FF4500")
        self.place_at_grid(cheetah, 'B1', scale_factor=0.6)
        self.play(FadeIn(cheetah))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        dot = Dot(color="#FF0000")
        self.place_at_grid(dot, 'B1', scale_factor=0.6)
        self.add(dot)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 120, 20], axis_config={"include_tip": False})
        graph = axes.plot(lambda x: 20 * x, x_range=[0, 5], color="#00FF00")
        group = VGroup(axes, graph)
        self.place_in_area(group, 'A3', 'E5', scale_factor=0.5)
        self.play(Create(group))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF00FF")
        # Ensure coordinates are mapped correctly using the axes
        secant = Line(axes.c2p(0,0), axes.c2p(5,100), color="#FF00FF")
        self.play(Create(secant))
        text_speed = Text("Average Speed = 20m/s", font_size=18, color="#00FF00")
        self.place_at_grid(text_speed, 'F3', scale_factor=0.8)
        self.play(Write(text_speed))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF8800")
        self.wait()
