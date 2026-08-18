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
        self.setup_layout("Prerequisite Physics: Conservation Laws", [
            "Energy and momentum are always conserved.",
            "Collisions preserve kinetic energy exactly.",
            "Phase space shows block velocities."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        
        vec1 = Arrow(start=ORIGIN, end=RIGHT*1.5, color=YELLOW)
        vec2 = Arrow(start=ORIGIN, end=LEFT*1.5, color=BLUE)
        momentum_group = VGroup(vec1, vec2).arrange(RIGHT)
        # Apply fix for issue 24/39: use B4-B6
        self.place_in_area(momentum_group, "B4", "B6", scale_factor=0.9)
        self.play(Create(momentum_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        
        # Using asset as requested in storyboard/issue 17
        energy_block1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg").scale(0.5)
        energy_block2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg").scale(0.5)
        circle1 = Circle(radius=0.5, color=GREEN, fill_opacity=0.5).surround(energy_block1)
        circle2 = Circle(radius=0.5, color=GREEN, fill_opacity=0.5).surround(energy_block2)
        energy_group = VGroup(VGroup(energy_block1, circle1), VGroup(energy_block2, circle2)).arrange(RIGHT, buff=0.5)
        
        # Apply fix for issue 25/40: use D4-D6
        self.place_in_area(energy_group, "D4", "D6", scale_factor=0.9)
        self.play(Create(energy_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        
        axes = Axes(x_range=[-1, 5, 1], y_range=[-1, 5, 1], axis_config={"include_tip": True})
        # Apply fix for issue 26/41: use E1-F6
        self.place_in_area(axes, "E1", "F6", scale_factor=0.5)
        self.play(Create(axes))
        self.wait(2)
