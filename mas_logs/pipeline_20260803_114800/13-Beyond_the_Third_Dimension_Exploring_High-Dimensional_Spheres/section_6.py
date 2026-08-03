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
        self.setup_layout("Real-World Echoes: Why It Matters", [
            "High-dimensional geometry shapes modern big data.",
            "These concepts refine error-correcting communication codes.",
            "Abstract math keeps our digital world connected."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        
        # Binary stream data
        binary_chars = "1011010010101100"
        binary_vgroup = VGroup(*[Text(char, font_size=24, color="#00FF00") for char in binary_chars])
        binary_vgroup.arrange(RIGHT, buff=0.3)
        
        # Adjusted position and scale based on VideoCritic feedback
        self.place_in_area(binary_vgroup, 'B1', 'B6', scale_factor=0.6)
        
        # Animation: binary stream appearing and "flowing"
        self.play(Write(binary_vgroup))
        self.play(binary_vgroup.animate.shift(LEFT * 0.5), rate_func=linear, run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        # Data points clustering into "spheres"
        points = VGroup(*[Dot(radius=0.05, color=BLUE_A) for _ in range(30)])
        # Randomly scatter points in C1-D6 area
        area_tl = self.grid["C1"]
        area_br = self.grid["D6"]
        for p in points:
            p.move_to([
                np.random.uniform(area_tl[0], area_br[0]),
                np.random.uniform(area_br[1], area_tl[1]),
                0
            ])
            
        self.play(FadeIn(points))
        
        # Define cluster centers
        center1 = self.grid["C2"]
        center2 = self.grid["D5"]
        
        # "Spheres of influence" (Circles)
        sphere1 = Circle(radius=0.7, color=BLUE, stroke_width=2).move_to(center1)
        sphere2 = Circle(radius=0.7, color=BLUE, stroke_width=2).move_to(center2)
        
        # Animations: Points move to clusters and spheres appear
        self.play(
            *[points[i].animate.move_to(center1 + np.random.normal(0, 0.25, 3) * [1,1,0]) for i in range(15)],
            *[points[i].animate.move_to(center2 + np.random.normal(0, 0.25, 3) * [1,1,0]) for i in range(15, 30)],
            Create(sphere1),
            Create(sphere2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        
        # Wi-Fi Icon Asset
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wifi.svg]
        wifi_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wifi.svg")
        wifi_icon.set_color("#00FFFF")
        
        # Adjusted position and scale based on VideoCritic feedback
        self.place_in_area(wifi_icon, 'E4', 'F5', scale_factor=0.7)
        
        # Fade out previous visuals and show Wi-Fi icon
        self.play(
            FadeOut(binary_vgroup),
            FadeOut(points),
            FadeOut(sphere1),
            FadeOut(sphere2),
            FadeIn(wifi_icon)
        )
        
        # Pulse animation (Scale 1.2 is 20% increase)
        self.play(
            wifi_icon.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=1
        )
        self.play(
            wifi_icon.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=1
        )
        
        self.wait(2)
        self.play(FadeOut(wifi_icon), FadeOut(self.title), FadeOut(self.lecture))
        self.wait(1)
