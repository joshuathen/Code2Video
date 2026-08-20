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
            "Disks move only between adjacent pegs in this variation.",
            "Moving three disks requires exactly twenty-six steps.",
            "The move count follows the pattern three to n minus one."
        ])
        
        # Prepare peg visualization using SVGMobjects
        peg_icon = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg"
        disk_icon = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        
        pegs = VGroup(*[SVGMobject(peg_icon, color=GRAY) for _ in range(3)])
        # Based on constraints: use D3, D4, D5 for tighter horizontal spread
        self.place_at_grid(pegs[0], "D3", scale_factor=0.7)
        self.place_at_grid(pegs[1], "D4", scale_factor=0.7)
        self.place_at_grid(pegs[2], "D5", scale_factor=0.7)
        
        # Connections between adjacent pegs
        line_ab = Line(pegs[0].get_right(), pegs[1].get_left(), color=WHITE)
        line_bc = Line(pegs[1].get_right(), pegs[2].get_left(), color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(pegs), Create(line_ab), Create(line_bc))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Represent disks as stacked SVGMobjects on peg A
        disks = VGroup(*[SVGMobject(disk_icon, color=BLUE) for _ in range(3)])
        # Stacked manually with slight offset above the peg
        for i, disk in enumerate(disks):
            disk.scale(0.5 - i * 0.05)
        disks.arrange(DOWN, buff=0.05).next_to(pegs[0], UP, buff=0.1)
        self.add(disks)
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        # Highlight constrained paths with yellow as requested in storyboard
        self.play(Indicate(line_ab, color="#FFFF00"), Indicate(line_bc, color="#FFFF00"))
        self.wait(2)
