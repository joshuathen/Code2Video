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
        self.setup_layout("Constrained Towers of Hanoi", [
            "Disks can only move between adjacent pegs.",
            "This constraint forbids direct jumping across pegs.",
            "Recursive movement follows a strict, path-limited graph."
        ])
        
        # Load assets
        disk_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disks.svg"
        
        # Setup rods and disks using SVG asset
        rod1 = Line(UP, DOWN).scale(0.5).set_color(WHITE)
        rod2 = Line(UP, DOWN).scale(0.5).set_color(WHITE)
        rod3 = Line(UP, DOWN).scale(0.5).set_color(WHITE)
        rods = VGroup(rod1, rod2, rod3).arrange(RIGHT, buff=0.8)
        
        # SVG icon representing disks
        disks = SVGMobject(disk_asset_path).set_color("#00FFFF")
        
        # Assembly
        tower_assembly = VGroup(rods, disks).arrange(DOWN, buff=0.5)
        self.place_in_area(tower_assembly, 'B3', 'E5', scale_factor=0.85)
        
        self.add(tower_assembly)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        # Flash restriction using asset
        self.play(disks.animate.set_color("#FF00FF"), run_time=0.5)
        self.play(disks.animate.set_color("#00FFFF"), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Show movement restriction (illegal path) using asset
        illegal_move = Arrow(start=rods[0].get_center(), end=rods[2].get_center(), color=RED)
        self.add(illegal_move)
        self.play(FadeOut(illegal_move), run_time=1.5)
        self.wait(2)
