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
            "Is the straight line actually the fastest path?",
            "The Cycloid is the true fastest curve.",
            "The Cycloid balances speed and distance perfectly."
        ]
        self.setup_layout("The Counter-Intuitive Truth: The Cycloid", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        
        # Visual: Show straight line vs curve
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 3, 1], axis_config={"include_tip": False}).scale(0.6)
        # Applying fix for Issue 27 and 29
        self.place_in_area(axes, "B3", "E6", scale_factor=0.7)
        self.add(axes)
        
        start = axes.c2p(0, 2)
        end = axes.c2p(3, 0)
        straight_line = Line(start, end, color=WHITE)
        self.add(straight_line)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        
        # Visual: Draw the cycloid
        # Cycloid parametric: x = r(t - sin(t)), y = r(1 - cos(t))
        r = 0.4
        cycloid = ParametricFunction(
            lambda t: axes.c2p(r*(t - np.sin(t)), -r*(1 - np.cos(t)) + 2),
            t_range=[0, 2*np.pi],
            color="#FF4500"
        )
        self.play(Create(cycloid))
        
        # Asset integration (Issue 18)
        particle_straight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg", color="#FF4500")
        particle_cycloid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg", color="#00FF00")
        
        particle_straight.scale(0.1)
        particle_cycloid.scale(0.1)
        
        particle_straight.move_to(start)
        particle_cycloid.move_to(start)
        
        self.add(particle_straight, particle_cycloid)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # Visual: Add label/highlight (Issue 28)
        label = Text("Cycloid Path", font_size=20, color="#FF4500")
        self.place_at_grid(label, "F4", scale_factor=0.7)
        self.play(Write(label))
        
        # Race animation
        self.play(
            MoveAlongPath(particle_straight, straight_line),
            MoveAlongPath(particle_cycloid, cycloid),
            run_time=3,
            rate_func=linear
        )
        
        self.wait(2)
