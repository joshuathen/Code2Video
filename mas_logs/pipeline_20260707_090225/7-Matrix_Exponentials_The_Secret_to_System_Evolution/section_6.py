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
        # Setup layout with specific title and lecture lines
        self.setup_layout(
            "Application: Solving Linear Systems", 
            [
                'Systems of linear differential equations use this tool.', 
                'The solution is e to At times the initial state.', 
                'Matrix exponentials are the fundamental propagators of linear systems.'
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), run_time=0.5)
        
        # Using Text instead of MathTex to avoid LaTeX dependency errors
        diff_eq = Text("dx/dt = Ax", font_size=32)
        initial_cond = Text("x(0)", color="#FFA500", font_size=28)
        
        # Positioning
        self.place_in_area(diff_eq, "A2", "B4")
        self.place_at_grid(initial_cond, "B5") # Fixed: closer to equation (Issue 48)
        
        self.play(FadeIn(diff_eq), FadeIn(initial_cond))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00"),
            run_time=0.5
        )
        
        # Create solution text using VGroup to allow targeted coloring/animation of specific parts
        sol_eq = VGroup(
            Text("x(t) = ", font_size=32),
            Text("e^At", font_size=32),
            Text(" x(0)", font_size=32)
        ).arrange(RIGHT, buff=0.1)
        
        # Fixed: Adjusted area to avoid propagator label collision (Issue 47)
        self.place_in_area(sol_eq, "C2", "C5", scale_factor=1.0)
        
        # Create propagator label
        propagator_label = Text("Propagator", font_size=24, color="#00FF00")
        self.place_at_grid(propagator_label, "D3")
        
        self.play(FadeIn(sol_eq))
        
        # Flash the term e^At and show label
        self.play(
            sol_eq[1].animate.set_color("#00FF00"),
            Indicate(sol_eq[1], color="#00FF00"),
            Write(propagator_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00"),
            run_time=0.5
        )

        # Define a vector field for the streamlines (stable spiral)
        def flow_func(p):
            x, y = p[:2]
            # Simple spiral sink dx/dt = -0.5x + y; dy/dt = -x - 0.5y
            return np.array([-0.5 * x + y, -x - 0.5 * y, 0])

        # Create StreamLines object
        stream_lines = StreamLines(
            flow_func, 
            x_range=[-2, 2], 
            y_range=[-2, 2], 
            stroke_width=2, 
            color=BLUE_E,
            opacity=0.4
        )
        # Fixed: Reduced width and scale to avoid cramped appearance (Issue 49)
        self.place_in_area(stream_lines, "E2", "F5", scale_factor=0.5)
        
        # Add 5 dots to follow streamlines
        start_points = [
            [1.5, 1.5, 0], 
            [-1.5, 1.0, 0], 
            [1.0, -1.5, 0], 
            [-1.2, -1.2, 0], 
            [0.5, 1.8, 0]
        ]
        
        dots = VGroup()
        for pt in start_points:
            dot = Dot(point=pt, color="#00FFFF", radius=0.06)
            dots.add(dot)
        
        # Scale and move dots group relative to stream_lines (Scale matches Issue 49)
        dots.scale(0.5)
        dots.move_to(stream_lines.get_center())
        
        def update_dot(mob, dt):
            # Calculate direction relative to center of field
            relative_pos = (mob.get_center() - stream_lines.get_center()) / 0.5
            v = flow_func(relative_pos)
            mob.shift(v * dt * 0.4) 
            # Recycle dots if they get too close to center
            if np.linalg.norm(relative_pos) < 0.1:
                mob.move_to(stream_lines.get_center() + np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), 0]) * 0.5)

        self.play(FadeIn(stream_lines))
        self.add(dots)
        
        # Attach updaters
        for dot in dots:
            dot.add_updater(update_dot)
            
        self.wait(4)
        
        # Clean up updaters
        for dot in dots:
            dot.remove_updater(update_dot)

        # Final state color correction
        self.play(self.lecture[2].animate.set_color(WHITE), run_time=0.5)
        self.wait(1)

# To render: manim -ql section_6.py Section6Scene
