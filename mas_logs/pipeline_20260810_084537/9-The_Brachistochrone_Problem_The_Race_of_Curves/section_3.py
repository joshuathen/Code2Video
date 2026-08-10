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
        lecture_lines = [
            "The optimal solution is a cycloid curve.",
            "It traces a point on a rolling wheel.",
            "Parametric equations define this elegant trajectory.",
            "Wheel rotation generates the path visually.",
            "Cycloids minimize travel time between points."
        ]
        self.setup_layout("The Mathematical Reveal: The Cycloid", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg]
        # Use SVGMobject for the wheel
        wheel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg", color=WHITE)
        point = Dot(color=RED)
        
        # Group them
        wheel_group = VGroup(wheel, point)
        # Applying critic fix 27: use place_in_area
        self.place_in_area(wheel_group, 'C3', 'D4', scale_factor=0.65)
        
        # Track the path
        path = VGroup()
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        eq = MathTex(r"x = r(\theta - \sin\theta), y = r(1 - \cos\theta)", font_size=24)
        # Applying critic fix 26: use place_at_grid B5
        self.place_at_grid(eq, 'B5', scale_factor=0.7)
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(ORANGE))
        
        # Rolling animation
        # Need to roll the wheel from left to right area
        radius = 0.5
        total_dist = 4.0
        
        def update_wheel(mob, dt):
            # Advance theta based on time
            theta = self.time * 2
            x = theta * radius
            
            # Position wheel base
            # Use original location from place_in_area as anchor
            base_pos = self.grid['C3'] + np.array([-1, 0, 0])
            mob.move_to(base_pos + np.array([x, 0, 0]))
            
            # Position point on rim (relative to wheel center)
            point.move_to(mob.get_center() + radius * np.array([np.sin(theta), -np.cos(theta), 0]))
            
            # Add trail
            new_dot = Dot(point.get_center(), radius=0.03, color=YELLOW)
            path.add(new_dot)
            self.add(new_dot)
            
            if x > total_dist:
                mob.remove_updater(update_wheel)

        wheel_group.add_updater(update_wheel)
        self.add(path)
        self.wait(4)
        wheel_group.remove_updater(update_wheel)
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(RED))
        self.wait(1)
