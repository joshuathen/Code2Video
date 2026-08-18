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
        lecture_lines = [
            "Lifeguards must reach drowning swimmers quickly.",
            "Running on sand is slower than swimming.",
            "A straight path is not always fastest.",
            "We seek the path of least time.",
            "Angle impacts total travel time."
        ]
        self.setup_layout("Introduction: The 'Life Guard' Problem", lecture_lines)

        # Assets
        beach = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/beach.svg", color="#F0E68C")
        ocean = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ocean.svg", color="#4682B4")
        lifeguard = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lifeguard.svg")
        swimmer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/swimmer.svg")
        
        # Visual Container
        boundary = Line(LEFT*2.5, RIGHT*2.5, color=WHITE)
        container = VGroup(beach, ocean, boundary)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(container, 'A4', 'F6', scale_factor=0.5)
        self.play(FadeIn(beach), FadeIn(ocean), Create(boundary), run_time=1)
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(lifeguard, 'B4', scale_factor=0.3)
        self.place_at_grid(swimmer, 'E6', scale_factor=0.3)
        self.play(FadeIn(lifeguard), FadeIn(swimmer), run_time=1)
        self.lecture[1].set_color(GOLD)

        # === Animation for Lecture Line 3 ===
        path_straight = Line(lifeguard.get_center(), swimmer.get_center(), color=WHITE)
        self.play(Create(path_straight), run_time=1.5)
        self.lecture[2].set_color(BLUE)

        # === Animation for Lecture Line 4 ===
        path_opt = VMobject()
        path_opt.set_points_smoothly([lifeguard.get_center(), self.grid['C5'], swimmer.get_center()])
        path_opt.set_color(GREEN)
        self.play(Create(path_opt), run_time=1.5)
        self.lecture[3].set_color(GREEN)

        # === Animation for Lecture Line 5 ===
        # Animated icon
        swimmer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/swimmer.svg").scale(0.2)
        swimmer_icon.move_to(self.grid['C5'])
        self.play(FadeIn(swimmer_icon), run_time=1)
        self.play(swimmer_icon.animate.move_to(self.grid['D5']), run_time=1.5)
        self.lecture[4].set_color(ORANGE)
        self.wait(1)
