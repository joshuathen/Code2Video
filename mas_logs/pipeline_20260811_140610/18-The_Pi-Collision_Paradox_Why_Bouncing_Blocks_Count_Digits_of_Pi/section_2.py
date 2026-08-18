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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Conservation of momentum governs elastic collisions.",
            "Conservation of kinetic energy limits block motion.",
            "Phase space visuals map these constraints perfectly."
        ]
        self.setup_layout("Prerequisite Setup: Momentum and Energy", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        momentum_eq = MathTex(r"p = m v", color="#FFFF00")
        block_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#FFFF00")
        
        # Align them in a VGroup to ensure they maintain hierarchy (B039)
        momentum_group = VGroup(block_icon, momentum_eq).arrange(RIGHT, buff=0.2)
        self.place_at_grid(momentum_group, 'B3', scale_factor=1.0) # Fixed per feedback
        
        self.play(FadeIn(momentum_group))
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        energy_eq = MathTex(r"K = \frac{1}{2} m v^2", color="#00FF00")
        self.place_at_grid(energy_eq, 'C3', scale_factor=1.0) # Fixed per feedback
        self.play(Write(energy_eq))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Simple Axes for Phase Space
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            axis_config={"include_tip": True, "color": "#808080"}
        )
        self.place_in_area(axes, 'D2', 'F5', scale_factor=0.5) # Fixed per feedback
        
        self.play(Create(axes))
        self.play(self.lecture[2].animate.set_color("#808080"))
        self.wait(2)
