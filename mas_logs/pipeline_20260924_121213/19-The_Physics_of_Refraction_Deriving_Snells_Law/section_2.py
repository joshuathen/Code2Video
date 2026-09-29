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
        lecture_lines = ["Light travels the fastest path.", "Velocity depends on refractive index.", "Least time path is optimal."]
        self.setup_layout("Prerequisite: Fermat’s Principle of Least Time", lecture_lines)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg]

        # === Animation for Lecture Line 1: Show path AB with laser icon ===
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        self.place_at_grid(laser, 'B4', scale_factor=0.5)
        path_ab = Line(start=self.grid["B4"], end=self.grid["E6"], color="#FFFF00")
        
        self.play(FadeIn(laser), Create(path_ab))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2: Animate path length minimization ===
        # Wrap animation group and place per issue 22/37
        anim_group = VGroup()
        curve = CurvedArrow(start_point=self.grid["B4"], end_point=self.grid["E6"], angle=PI/6, color="#00FF00")
        anim_group.add(curve)
        self.place_in_area(anim_group, 'C3', 'F6', scale_factor=0.6)
        
        self.play(ReplacementTransform(path_ab, anim_group))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3: Illustrate refraction with prism icon ===
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        self.place_at_grid(prism, 'D3', scale_factor=0.5)
        boundary = Line(self.grid["D2"], self.grid["D6"], color=GRAY)
        normal = DashedLine(self.grid["B4"], self.grid["F4"], color=GRAY)
        incident = Line(self.grid["B4"], self.grid["D4"], color=WHITE)
        refracted = Line(self.grid["D4"], self.grid["E5"], color=WHITE)
        
        # Combine into group to place in area as requested by issue 22/37
        refraction_group = VGroup(boundary, normal, incident, refracted, prism)
        self.place_in_area(refraction_group, 'C3', 'F6', scale_factor=0.6)
        
        self.play(Create(refraction_group))
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
