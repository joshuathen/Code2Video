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
            "We add a strict adjacent move constraint.",
            "Disks move only to immediate neighboring pegs.",
            "Three disks require three-cubed minus one steps."
        ]
        self.setup_layout("The Constrained Towers of Hanoi", lecture_lines)
        
        # Define visual assets
        # Using SVG files as specified
        peg_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg"
        disk_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        towers = VGroup(*[SVGMobject(peg_path).set_color(YELLOW) for _ in range(3)])
        # Fix: Towers alignment and scale based on criticism
        self.place_in_area(towers, 'B2', 'C6', scale_factor=0.6)
        
        disks = VGroup(*[
            SVGMobject(disk_path).set_color("#FF00FF")
            for _ in range(3)
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(Create(towers))
        
        # Position disks on the first tower (tower[0])
        for i, disk in enumerate(disks):
            disk.scale(0.8 - i*0.1) # vary sizes
            disk.next_to(towers[0], DOWN, buff=-0.5 - i*0.3)
        self.play(FadeIn(disks))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        formula = MathTex("3^3 - 1 = 26", color=WHITE)
        # Fix: Formula placement
        self.place_at_grid(formula, 'D4', scale_factor=1.0)
        self.play(Write(formula))
        self.wait(2)
