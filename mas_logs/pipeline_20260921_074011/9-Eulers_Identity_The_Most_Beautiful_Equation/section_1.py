from manim import *
import os

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
        self.setup_layout("Euler’s Identity: The Most Beautiful Equation", [
            "e, i, and pi bridge separate math worlds.",
            "We aim to unify them.",
            "They create Euler's beautiful identity."
        ])
        
        # Animations
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        # Load asset
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        self.place_at_grid(bridge_icon, 'A3', scale_factor=0.5)
        
        title_intro = Text("Bridging the Worlds", font_size=24, color="#FFFFFF")
        self.place_at_grid(title_intro, 'A4')
        self.play(FadeIn(bridge_icon), FadeIn(title_intro))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        axes = Axes(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1], 
            axis_config={"include_tip": True}
        )
        # Apply fix for issue 20/35
        self.place_in_area(axes, 'B2', 'D4', scale_factor=0.5)
        self.play(Create(axes))
        
        circle = Circle(radius=1.0, color="#00FFFF")
        self.place_in_area(circle, 'B2', 'D4', scale_factor=0.5)
        self.play(Create(circle))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        dot = Dot(color="#FFFF00")
        # Apply fix for issue 22/37
        self.place_at_grid(dot, 'C3', scale_factor=0.8)
        label_z = MathTex("z", color="#FFFF00")
        label_z.next_to(dot, UP)
        
        self.play(FadeIn(dot), Write(label_z))
        
        # Rotating point
        theta = ValueTracker(0)
        # Fix: Need to ensure dot tracks circular path correctly
        dot.add_updater(lambda d: d.move_to(axes.c2p(np.cos(theta.get_value()), np.sin(theta.get_value()))))
        label_z.add_updater(lambda l: l.next_to(dot, UP))
        
        self.play(theta.animate.set_value(2 * PI), run_time=3, rate_func=linear)
        
        # Identity
        identity = MathTex("e^{i\\theta} = \\cos\\theta + i\\sin\\theta", color="#FF0000")
        # Apply fix for issue 21/36
        self.place_at_grid(identity, 'E2', scale_factor=0.9)
        self.play(Write(identity))
        self.wait(2)
