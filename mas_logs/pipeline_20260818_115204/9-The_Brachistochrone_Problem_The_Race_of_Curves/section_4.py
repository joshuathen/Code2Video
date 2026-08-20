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
            "The optimal path is called a cycloid.",
            "It is traced by a point on rolling wheels.",
            "This curve perfectly balances distance and speed.",
            "Steeper starts build velocity for the end.",
            "The cycloid is the true fastest path."
        ]
        self.setup_layout("The Solution: The Cycloid", lecture_lines)

        # Assets
        wheel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wheel.svg", color="#FFFFFF")
        bicycle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bicycle.svg", color="#00FF00")
        
        r = 0.5
        point = Dot(color="#FF0000", radius=0.08)
        
        # Cycloid tracing
        def get_cycloid_point(t):
            return np.array([r * (t - np.sin(t)), -r * (1 - np.cos(t)), 0])
        
        curve = ParametricFunction(
            lambda t: get_cycloid_point(t),
            t_range=[0, 2 * PI],
            color="#00FF00"
        )
        
        cycloid_anim = VGroup(wheel, curve, point)
        # Apply Issue 30/45 fix
        self.place_in_area(cycloid_anim, 'B2', 'D5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(curve), FadeIn(wheel))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        
        # Rolling circle
        wheel.move_to(self.grid["B3"] + np.array([-r*2*PI, 0, 0]))
        point.move_to(wheel.get_center() + np.array([0, -r, 0]))
        self.add(wheel, point)
        
        def update_circle(m, dt):
            t = self.time * 2
            if t > 2 * PI: t = 2 * PI
            m.move_to(self.grid["B3"] + np.array([r * (t - 2 * PI), 0, 0]))
        
        def update_point(m, dt):
            t = self.time * 2
            if t > 2 * PI: t = 2 * PI
            m.move_to(self.grid["B3"] + get_cycloid_point(t))
            
        wheel.add_updater(update_circle)
        point.add_updater(update_point)
        self.wait(4)
        wheel.remove_updater(update_circle)
        point.remove_updater(update_point)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        # Verify cycloid with bicycle
        self.place_at_grid(bicycle, 'E5', scale_factor=0.8)
        self.play(FadeIn(bicycle))
        self.play(Indicate(curve))
        self.wait(2)
