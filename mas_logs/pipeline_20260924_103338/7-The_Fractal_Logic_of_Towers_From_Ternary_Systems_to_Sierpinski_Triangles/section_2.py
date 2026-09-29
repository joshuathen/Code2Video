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
        lecture_lines = ["Constraint limits movement to adjacent pegs only.", "Disks hop between A, B, and C.", "This structure forms a cyclic ternary path."]
        self.setup_layout("Constrained Towers of Hanoi", lecture_lines)
        
        # --- Pre-calculate elements ---
        # Assets as per storyboard: tower.svg, disc.svg, peg.svg
        # Using SVG placeholders as the path needs to be loaded as SVGMobjects
        tower = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg")
        disc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disc.svg")
        peg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg")
        
        # --- Animation for Lecture Line 1 ---
        # Display tower with discs on peg A, center in Row E. Color: #FF4500.
        self.lecture[0].set_color("#FF4500")
        group = VGroup(tower, disc, peg)
        self.place_in_area(group, "B3", "D5", scale_factor=0.7)
        self.play(FadeIn(group))
        self.wait(1)

        # --- Animation for Lecture Line 2 ---
        # Iterate disc moves through pegs A-B-C. Color: #1E90FF.
        self.lecture[1].set_color("#1E90FF")
        # Movement simulation
        self.play(group.animate.move_to(self.grid["B5"]))
        self.wait(0.5)
        self.play(group.animate.move_to(self.grid["C5"]))
        self.wait(1)

        # --- Animation for Lecture Line 3 ---
        # Display flashing red 'X' over restricted non-adjacent move near peg. Color: #FF0000.
        self.lecture[2].set_color("#FF0000")
        warning = Tex("X", font_size=72, color=RED)
        # Positioned at B5 as requested in recent issues
        self.place_at_grid(warning, "B5", scale_factor=0.5)
        self.play(Flash(warning, color=RED, line_length=0.2))
        self.play(FadeIn(warning))
        self.play(Indicate(warning))
        self.wait(2)
