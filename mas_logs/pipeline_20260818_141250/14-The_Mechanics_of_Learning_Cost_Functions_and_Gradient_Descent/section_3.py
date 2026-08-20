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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Gradient descent is our navigation tool.",
            "The derivative tells us the steepest path.",
            "We follow the slope to reach minima."
        ]
        self.setup_layout("The Descent Strategy: Gradient Descent", lecture_lines)
        
        # Define objects
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/terrain.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg]
        
        terrain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/terrain.svg", color=WHITE)
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color="#00CCFF")
        mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg", color=WHITE)
        
        axes = Axes(x_range=[-2, 2], y_range=[-1, 3], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: x**2, x_range=[-1.5, 1.5], color=WHITE)
        
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.6)
        self.place_in_area(curve, 'B3', 'F6', scale_factor=0.6)
        
        dot = Dot(color=YELLOW)
        dot.move_to(axes.c2p(1, 1))
        
        tangent_line = Line(start=LEFT, end=RIGHT, color=RED).set_length(1)
        slope_arrow = compass
        
        # Setup initial state
        self.add(axes, curve, dot)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        title = Text("Gradient Descent: Slope", font_size=24, color="#FFFFFF")
        self.place_at_grid(title, 'A2', scale_factor=0.9)
        self.add(title)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        # Add terrain asset
        self.place_at_grid(terrain, "B1", scale_factor=0.3)
        self.add(terrain)
        # Position tangent
        tangent_line.rotate(0.8).move_to(dot.get_center())
        self.add(tangent_line)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00CCFF"))
        # Add compass asset
        self.place_at_grid(slope_arrow, "C1", scale_factor=0.3)
        # Add mountain asset
        self.place_at_grid(mountain, "D1", scale_factor=0.3)
        self.add(slope_arrow, mountain)
        
        # Animate movement
        self.play(dot.animate.move_to(axes.c2p(0, 0)), run_time=2)
        
        self.wait(1)
