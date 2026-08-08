from manim import *

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
        lecture_lines = ["PDEs require boundary conditions for uniqueness.", "String length defines the boundary state.", "Plucking the string is the initial condition."]
        self.setup_layout("Boundary and Initial Conditions", lecture_lines)
        
        # Create objects
        axes = Axes(x_range=[0, 4, 1], y_range=[-1.5, 1.5, 1], axis_config={"include_numbers": False})
        curve = axes.plot(lambda x: np.sin(np.pi * x / 4), x_range=[0, 4], color=YELLOW)
        
        # Asset: rod.svg for marking boundaries
        rod_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg")
        rod_left = rod_asset.copy().set_color(RED)
        rod_right = rod_asset.copy().set_color(RED)
        
        # Place axes and rods
        self.place_in_area(axes, "B4", "F6", scale_factor=0.5)
        
        # Use axes to position rods at boundary points (0 and 4 on x-axis)
        rod_left.move_to(axes.c2p(0, 0))
        rod_right.move_to(axes.c2p(4, 0))
        
        # Initially display
        self.add(axes, curve, rod_left, rod_right)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(Indicate(rod_left), Indicate(rod_right))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        # Initial condition animation
        self.play(curve.animate.shift(UP * 0.5), run_time=1)
        self.play(curve.animate.shift(DOWN * 0.5), run_time=1)
        self.wait(2)
