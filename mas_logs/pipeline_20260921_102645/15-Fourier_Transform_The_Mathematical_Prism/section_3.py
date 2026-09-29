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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Core Concept: The Rotating Windmill", [
            "Fourier Transform acts as a mathematical sieve.", 
            "Multiply signal by complex rotating exponentials.", 
            "Align rotation speed with signal frequency.", 
            "Constructive interference creates spectrum peaks.", 
            "Shift center of gravity reveals components."
        ])
        
        # Load asset
        windmill_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg")
        
        # Setup Animation Objects
        circle = Circle(radius=0.8, color=GRAY)
        arm = Line(ORIGIN, 0.8 * RIGHT, color="#FFA500")
        tip = Dot(arm.get_end(), color="#FFA500")
        shadow = Dot(np.array([0.8, 0, 0]), color=BLUE) # Projection on Y axis
        
        windmill = VGroup(circle, arm, tip, windmill_svg)
        windmill_svg.scale(0.5).move_to(circle.get_center())
        
        # Applying requested layout updates
        self.place_in_area(windmill, 'C2', 'D3', scale_factor=1.2)
        self.place_in_area(shadow, 'C4', 'D5', scale_factor=1.2)
        
        # Update trackers
        angle = ValueTracker(0)
        
        def update_arm(m):
            c = circle.get_center()
            theta = angle.get_value()
            m.put_start_and_end_on(c, c + 0.8 * np.array([np.cos(theta), np.sin(theta), 0]))
        
        arm.add_updater(update_arm)
        tip.add_updater(lambda m: m.move_to(arm.get_end()))
        shadow.add_updater(lambda m: m.move_to(np.array([circle.get_center()[0], circle.get_center()[1] + 0.8 * np.sin(angle.get_value()), 0])))

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(windmill), FadeIn(shadow))
        self.lecture[0].set_color("#FFA500")

        # === Animation for Lecture Line 2 ===
        self.play(angle.animate.set_value(2*PI), run_time=2)
        self.lecture[1].set_color("#FFA500")

        # === Animation for Lecture Line 3 ===
        self.play(angle.animate.set_value(4*PI), run_time=2)
        self.lecture[2].set_color("#FFA500")

        # === Animation for Lecture Line 4 ===
        self.play(angle.animate.set_value(6*PI), run_time=2)
        self.lecture[3].set_color("#FFA500")

        # === Animation for Lecture Line 5 ===
        self.play(angle.animate.set_value(8*PI), run_time=2)
        self.lecture[4].set_color("#FFA500")

        self.wait(2)
