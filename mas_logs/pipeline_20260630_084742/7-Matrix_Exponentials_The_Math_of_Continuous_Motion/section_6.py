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
        lecture_lines = [
            'Matrix exponentials solve systems of linear differential equations.',
            'The term e to the At evolves the initial state.',
            'It acts as a flow, moving systems through time.'
        ]
        self.setup_layout("Application: Solving Dynamic Systems", lecture_lines)
        
        # Colors
        L1_COLOR = "#00FFFF" # Cyan
        L2_COLOR = "#FFFF00" # Yellow
        L3_COLOR = "#00FF00" # Green

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(L1_COLOR), run_time=0.5)
        
        # Solution formula - Using Text instead of MathTex to avoid LaTeX dependency error
        formula = Text(
            "d/dt u(t) = Au(t)  =>  u(t) = e^(At) u(0)",
            color=WHITE,
            font_size=24
        )
        self.place_in_area(formula, "A2", "B5", scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(L2_COLOR), run_time=0.5)
        
        # Particles setup
        num_particles = 15
        np.random.seed(10)
        # Distribution around u(0)
        initial_offsets = np.random.uniform(-0.8, 0.8, (num_particles, 3))
        initial_offsets[:, 2] = 0
        
        flow_center = self.grid["D4"]
        particles = VGroup(*[
            Dot(point=flow_center + pos, radius=0.06, color=L2_COLOR) 
            for pos in initial_offsets
        ])
        
        # Using Text instead of MathTex to avoid LaTeX dependency error
        u0_label = Text("u(0)", color=L2_COLOR, font_size=24)
        self.place_at_grid(u0_label, "C2", scale_factor=0.7)
        
        self.play(
            LaggedStart(*[FadeIn(p, scale=0.5) for p in particles], lag_ratio=0.05),
            FadeIn(u0_label)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(L3_COLOR), run_time=0.5)
        
        # Flow logic
        # Matrix exponential e^{At} for A = [[-0.2, -2], [2, -0.2]]
        t_tracker = ValueTracker(0)
        
        def get_flow_pos(start_world_pos, t):
            rel_pos = start_world_pos - flow_center
            # Exponential of spiral matrix
            decay = np.exp(-0.15 * t)
            angle = t * 1.2
            rot = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0, 0, 1]
            ])
            return flow_center + decay * np.dot(rot, rel_pos)

        # Slider
        slider_line = Line(self.grid["F2"], self.grid["F5"], color=WHITE, stroke_width=2)
        slider_dot = Dot(color=L3_COLOR).move_to(slider_line.get_start())
        slider_label = Text("Time (t)", font_size=18).next_to(slider_line, DOWN, buff=0.1)
        
        self.add(slider_line, slider_dot, slider_label)
        
        # Link slider to tracker
        slider_dot.add_updater(lambda d: d.move_to(
            slider_line.point_from_proportion(t_tracker.get_value() / 4.0)
        ))
        
        # Link particles to tracker
        for p in particles:
            # Persistent starting position closure
            p.start_pos = p.get_center().copy()
            p.add_updater(lambda m, tracker=t_tracker: m.move_to(
                get_flow_pos(m.start_pos, tracker.get_value())
            ))

        # Flow animation
        self.play(t_tracker.animate.set_value(4.0), run_time=5, rate_func=linear)
        self.wait(2)
